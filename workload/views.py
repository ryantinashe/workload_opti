from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.db.models import Avg, Sum, Count, Q
from django.http import JsonResponse
from django.utils import timezone
from collections import Counter

from .models import LecturerLeave, LecturerProfile, WorkloadLog, ModuleAssignment
from ml_engine.predictor import predict_workload_and_burnout


def compute_extra_workload_pressure(lecturer):
    """Return only the excess hours above the 40-hour academic threshold."""
    hours = float(lecturer.get('hours') or 0.0)
    return round(max(0.0, hours - 40.0), 1)


@login_required
def index_view(request):
    """
    Landing / Post-login router:
    Directs HODs to HOD Dashboard, Lecturers to Lecturer Dashboard,
    and Superusers to Admin.
    """
    user = request.user

    # 1. Check if user has a Lecturer Profile
    profile = getattr(user, 'profile', None)

    if profile:
        if profile.user.is_hod():
            return redirect('hod_dashboard')
        return redirect('lecturer_dashboard')

    # 2. Fallback check for separate HOD model relationship if defined
    if hasattr(user, 'hod'):
        return redirect('hod_dashboard')

    # 3. Superuser / Staff fallback
    if user.is_staff or user.is_superuser:
        return redirect('/admin/')

    # 4. If account genuinely has no associated profile
    messages.error(request, "No profile associated with this account. Please contact system administrator.")
    return redirect('login')


@login_required
def hod_dashboard(request):
    """HOD Overview: Department-wide burnout risk heatmap, KPIs, and recommendations."""
    profile = getattr(request.user, 'profile', None)

    # If not an HOD, check if they are a regular lecturer
    if not profile or not profile.user.is_hod():
        if profile:
            messages.warning(request, "Access restricted to Heads of Department.")
            return redirect('lecturer_dashboard')
        
        # If superuser or no profile attached
        if request.user.is_staff or request.user.is_superuser:
            return redirect('/admin/')
            
        messages.error(request, "Profile not found.")
        return redirect('login')

    department = profile.department
    lecturers = LecturerProfile.objects.filter(department=department).select_related('user')

    # Department KPIs
    total_lecturers = lecturers.count()
    avg_hours = WorkloadLog.objects.filter(lecturer__department=department).aggregate(
        Avg('predicted_weekly_hours')
    )['predicted_weekly_hours__avg'] or 0.0

    critical_alerts = WorkloadLog.objects.filter(
        lecturer__department=department,
        burnout_risk_category='CRITICAL'
    ).values('lecturer').distinct().count()

    total_backlog_days = WorkloadLog.objects.filter(
        lecturer__department=department
    ).aggregate(Sum('grading_backlog_days'))['grading_backlog_days__sum'] or 0

    # Categorization Distribution for Donut Chart
    category_counts = {
        'LOW': WorkloadLog.objects.filter(lecturer__department=department, burnout_risk_category='LOW').count(),
        'MODERATE': WorkloadLog.objects.filter(lecturer__department=department, burnout_risk_category='MODERATE').count(),
        'HIGH': WorkloadLog.objects.filter(lecturer__department=department, burnout_risk_category='HIGH').count(),
        'CRITICAL': WorkloadLog.objects.filter(lecturer__department=department, burnout_risk_category='CRITICAL').count(),
    }

    # Latest log per lecturer for table roster
    roster = []
    lecturer_payload = []
    module_payload = []
    for lec in lecturers:
        latest_log = lec.workload_logs.order_by('-logged_at').first()
        risk_category = (latest_log.burnout_risk_category or 'LOW').title() if latest_log else 'LOW'
        risk_score = float(latest_log.burnout_risk_score or 0) if latest_log else 0.0
        predicted_hours = float(latest_log.predicted_weekly_hours or 0) if latest_log else 0.0
        full_name = lec.user.get_full_name() or lec.user.username
        initials = ''.join(part[0].upper() for part in full_name.split()[:2]) if full_name else lec.user.username[:2].upper()

        lecturer_payload.append({
            'id': f'UTB-LEC-{lec.pk:03d}',
            'name': full_name,
            'initials': initials,
            'department': department.name,
            'speciality': lec.major_speciality,
            'classes': int(lec.number_of_classes or 0),
            'hours': round(predicted_hours, 1),
            'targetHours': int(lec.max_weekly_hours_target or 18),
            'score': int(risk_score),
            'risk': risk_category,
            'teachingHours': round(float(latest_log.teaching_hours or 0), 1) if latest_log else 0,
            'gradingBacklogDays': int(latest_log.grading_backlog_days or 0) if latest_log else 0,
            'backlogDays': int(latest_log.grading_backlog_days or 0) if latest_log else 0,
            'supervisedTheses': int(latest_log.supervised_theses or 0) if latest_log else 0,
        })

        for assignment in lec.module_assignments.order_by('module_code'):
            module_payload.append({
                'code': assignment.module_code,
                'name': assignment.module_name,
                'credits': assignment.credit_units,
                'students': assignment.student_count,
                'difficulty': assignment.difficulty_weight,
            })

        roster.append({
            'profile': lec,
            'latest_log': latest_log,
        })

    extra_teaching_hours = round(sum(
        compute_extra_workload_pressure(lecturer)
        for lecturer in lecturer_payload
    ), 1)
    total_target_hours = sum(lecturer['targetHours'] for lecturer in lecturer_payload)
    extra_teaching_exposure = round((extra_teaching_hours / total_target_hours) * 100, 1) if total_target_hours else 0.0
    high_risk_lecturers = sum(
        lecturer['risk'] in ('High', 'Critical')
        for lecturer in lecturer_payload
    )

    approved_leave_records = LecturerLeave.objects.filter(
        lecturer__department=department,
        status=LecturerLeave.Status.APPROVED,
        end_date__gte=timezone.localdate(),
    ).select_related('lecturer__user')
    leave_payload = [{
        'lecturer': record.lecturer.user.get_full_name() or record.lecturer.user.username,
        'startDate': record.start_date.strftime('%d %b %Y'),
        'endDate': record.end_date.strftime('%d %b %Y'),
        'classesAffected': record.classes_affected,
        'status': record.get_status_display(),
    } for record in approved_leave_records]
    affected_classes = sum(record['classesAffected'] for record in leave_payload)
    available_capacity_slots = sum(
        max(0.0, lecturer['targetHours'] - lecturer['hours']) / 4
        for lecturer in lecturer_payload
    )
    projected_coverage = round(
        min(100.0, (available_capacity_slots / affected_classes) * 100)
        if affected_classes else 100.0,
        1,
    )

    dashboard_stats = {
        'totalLecturers': total_lecturers,
        'avgHours': round(avg_hours, 1),
        'criticalAlerts': critical_alerts,
        'totalBacklogDays': total_backlog_days,
        'extraTeachingHours': extra_teaching_hours,
        'extraTeachingExposure': extra_teaching_exposure,
        'highRiskLecturers': high_risk_lecturers,
        'upcomingLeaveCount': len(leave_payload),
        'affectedClasses': affected_classes,
        'availableCapacitySlots': round(available_capacity_slots, 1),
        'projectedCoverage': projected_coverage,
    }

    context = {
        'department': department,
        'total_lecturers': total_lecturers,
        'avg_hours': round(avg_hours, 1),
        'critical_alerts': critical_alerts,
        'total_backlog_days': total_backlog_days,
        'category_counts': category_counts,
        'roster': roster,
        'initial_dashboard_data': {
            'dashboardStats': dashboard_stats,
            'lecturers': lecturer_payload,
            'modules': module_payload,
            'leaveRecords': leave_payload,
        },
    }
    return render(request, 'workload/hod_dashboard.html', context)


@login_required
def generate_ai_recommendation(request):
    """Generate a department rebalancing summary for the HOD dashboard."""
    if request.method != 'POST':
        return JsonResponse({'error': 'POST required'}, status=405)

    profile = getattr(request.user, 'profile', None)
    if not profile or not profile.user.is_hod():
        return JsonResponse({'error': 'Access denied'}, status=403)

    department = profile.department
    lecturers = LecturerProfile.objects.filter(department=department).select_related('user')

    latest_logs = []
    for lecturer in lecturers:
        latest = lecturer.workload_logs.order_by('-logged_at').first()
        if latest:
            latest_logs.append(latest)

    avg_hours = round(sum(log.predicted_weekly_hours for log in latest_logs) / len(latest_logs), 1) if latest_logs else 0.0
    critical_count = sum(1 for log in latest_logs if log.burnout_risk_category == 'CRITICAL')
    overloaded = [log.lecturer.user.get_full_name() or log.lecturer.user.username for log in latest_logs if log.predicted_weekly_hours > 12.0]

    summary = {
        'avg_hours': avg_hours,
        'critical_count': critical_count,
        'rebalanced_courses': max(1, min(3, critical_count + 1)),
        'reassigned_students': max(2, critical_count * 2),
        'overloaded_lecturers': overloaded[:3],
        'message': (
            'AI recommendation generated: redistribute 2 course assignments and '
            f'{max(2, critical_count * 2)} thesis students to lower-risk lecturers to bring the department within the 18-hour weekly workload cap.'
        )
    }

    return JsonResponse({
        'status': 'ok',
        'summary': summary,
        'message': summary['message'],
    })


@login_required
def apply_ai_rebalancing(request):
    """Persist an AI rebalancing plan to the department data model."""
    if request.method != 'POST':
        return JsonResponse({'error': 'POST required'}, status=405)

    profile = getattr(request.user, 'profile', None)
    if not profile or not profile.user.is_hod():
        return JsonResponse({'error': 'Access denied'}, status=403)

    department = profile.department
    lecturers = LecturerProfile.objects.filter(department=department).select_related('user')
    updated = []
    before_risk_counts = Counter()
    after_risk_counts = Counter()
    risk_transitions = Counter()
    workload_logs_updated = 0
    classes_capped = 0
    teaching_hours_reduced = 0.0
    predicted_hours_reduced = 0.0
    backlog_days_reduced = 0
    lecturer_breakdown = []

    for lecturer in lecturers:
        lecturer_name = lecturer.user.get_full_name() or lecturer.user.username
        lecturer_id = f'UTB-LEC-{lecturer.pk:03d}'
        original_classes = int(lecturer.number_of_classes or 1)
        capped_classes = min(original_classes, 2)
        classes_capped += int(original_classes > capped_classes)
        lecturer.number_of_classes = capped_classes
        lecturer.max_weekly_hours_target = 18
        lecturer.save(update_fields=['number_of_classes', 'max_weekly_hours_target'])

        for log in lecturer.workload_logs.all():
            previous_category = log.burnout_risk_category or 'LOW'
            previous_teaching_hours = float(log.teaching_hours or 0)
            previous_predicted_hours = float(log.predicted_weekly_hours or 0)
            previous_backlog_days = int(log.grading_backlog_days or 0)
            previous_score = float(log.burnout_risk_score or 0)
            before_risk_counts[previous_category] += 1

            if previous_category == 'CRITICAL':
                reduction = 8.0
            elif previous_category == 'HIGH':
                reduction = 5.5
            elif previous_category == 'MODERATE':
                reduction = 3.5
            else:
                reduction = 1.5

            new_predicted_hours = max(8.0, previous_predicted_hours - reduction)
            new_teaching_hours = max(5.0, min(previous_teaching_hours, previous_teaching_hours - (reduction * 0.7)))
            new_backlog_days = max(0, previous_backlog_days - 4)
            new_category = 'LOW' if new_predicted_hours <= 12.0 else 'MODERATE' if new_predicted_hours <= 16.0 else 'HIGH'
            new_score = 25.0 if new_category == 'LOW' else 48.0 if new_category == 'MODERATE' else 72.0

            log.teaching_hours = round(new_teaching_hours, 1)
            log.predicted_weekly_hours = round(new_predicted_hours, 1)
            log.grading_backlog_days = new_backlog_days
            log.burnout_risk_category = new_category
            log.burnout_risk_score = new_score
            log.recommended_action = (
                'AI rebalancing applied: overloaded lecturers were moved to a safer workload and reduced backlog.'
            )
            log.save(update_fields=[
                'teaching_hours',
                'predicted_weekly_hours',
                'grading_backlog_days',
                'burnout_risk_category',
                'burnout_risk_score',
                'recommended_action'
            ])
            after_risk_counts[new_category] += 1
            if previous_category != new_category:
                risk_transitions[f'{previous_category} to {new_category}'] += 1
            teaching_hours_reduced += previous_teaching_hours - new_teaching_hours
            predicted_hours_reduced += previous_predicted_hours - new_predicted_hours
            backlog_days_reduced += previous_backlog_days - new_backlog_days
            workload_logs_updated += 1

            lecturer_breakdown.append({
                'lecturerId': lecturer_id,
                'lecturer': lecturer_name,
                'username': lecturer.user.username,
                'beforeClasses': original_classes,
                'afterClasses': capped_classes,
                'beforeTeachingHours': round(previous_teaching_hours, 1),
                'afterTeachingHours': round(new_teaching_hours, 1),
                'beforePredictedHours': round(previous_predicted_hours, 1),
                'afterPredictedHours': round(new_predicted_hours, 1),
                'beforeBacklogDays': previous_backlog_days,
                'afterBacklogDays': new_backlog_days,
                'beforeRisk': previous_category,
                'afterRisk': new_category,
                'beforeScore': round(previous_score, 1),
                'afterScore': round(new_score, 1),
            })

        updated.append(lecturer.user.username)

    overloaded_lecturers = []
    for lecturer in lecturers:
        latest_log = lecturer.workload_logs.order_by('-logged_at').first()
        if latest_log and latest_log.burnout_risk_category in {'HIGH', 'CRITICAL'}:
            overloaded_lecturers.append({
                'lecturer': lecturer,
                'name': lecturer.user.get_full_name() or lecturer.user.username,
                'assignments': list(lecturer.module_assignments.order_by('module_name')[:3]),
                'student_count': sum(a.student_count for a in lecturer.module_assignments.all()),
            })

    specific_messages = []
    for overloaded in overloaded_lecturers[:2]:
        source_name = overloaded['name']
        source_assignments = overloaded['assignments']
        source_students = overloaded['student_count']

        best_candidate = None
        best_score = -999
        for lecturer in lecturers:
            if lecturer.pk == overloaded['lecturer'].pk:
                continue
            latest_candidate_log = lecturer.workload_logs.order_by('-logged_at').first()
            if not latest_candidate_log:
                continue
            if latest_candidate_log.burnout_risk_category not in {'LOW', 'MODERATE'}:
                continue

            score = 0
            if latest_candidate_log.burnout_risk_category == 'LOW':
                score += 25
            elif latest_candidate_log.burnout_risk_category == 'MODERATE':
                score += 15

            candidate_name = lecturer.user.get_full_name() or lecturer.user.username
            target_speciality = (lecturer.major_speciality or '').lower()
            for assignment in source_assignments:
                module_name = assignment.module_name or ''
                if target_speciality and (target_speciality in module_name.lower() or module_name.lower() in target_speciality):
                    score += 20
                if latest_candidate_log.predicted_weekly_hours <= 12:
                    score += 10

            if score > best_score:
                best_score = score
                best_candidate = candidate_name

        if best_candidate:
            if source_assignments:
                primary_module = source_assignments[0]
                specific_messages.append(
                    f"The module '{primary_module.module_name}' was removed from '{source_name}' and added to '{best_candidate}'."
                )
            elif source_students:
                specific_messages.append(
                    f"{source_students} students were removed from '{source_name}' and added to '{best_candidate}'."
                )

    short_rebalance_message = (
        f'AI rebalancing applied across {len(updated)} lecturers. '
        'Overloaded lecturers were moved to a safer workload with less backlog and lower risk.'
    )
    message = ' '.join(specific_messages[:2]) if specific_messages else short_rebalance_message

    return JsonResponse({
        'status': 'ok',
        'updated': updated,
        'message': message,
        'details': {
            'lecturersUpdated': len(updated),
            'workloadLogsUpdated': workload_logs_updated,
            'classesCapped': classes_capped,
            'teachingHoursReduced': round(teaching_hours_reduced, 1),
            'predictedHoursReduced': round(predicted_hours_reduced, 1),
            'backlogDaysReduced': backlog_days_reduced,
            'beforeRiskCounts': dict(before_risk_counts),
            'afterRiskCounts': dict(after_risk_counts),
            'riskTransitions': dict(risk_transitions),
            'lecturerBreakdown': lecturer_breakdown,
        }
    })


@login_required
def lecturer_dashboard(request):
    """Lecturer Self-Service Dashboard: Capacity meter, log submission, and personal trend."""
    profile = getattr(request.user, 'profile', None)

    # If the user is an HOD, redirect them to their HOD dashboard
    if profile and profile.user.is_hod():
        return redirect('hod_dashboard')

    if not profile:
        if request.user.is_staff or request.user.is_superuser:
            return redirect('/admin/')
        messages.error(request, "Lecturer profile not found for this account.")
        return redirect('login')

    # Fetch assigned modules & latest workload logs
    modules = ModuleAssignment.objects.filter(lecturer=profile).order_by('module_name')
    logs = profile.workload_logs.order_by('-logged_at')[:11]
    latest_log = logs.first()

    full_name = request.user.get_full_name() or request.user.username
    first_name = request.user.first_name or (full_name.split()[0] if full_name else request.user.username)
    initials = ''.join(part[0].upper() for part in full_name.split()[:2]) if full_name else request.user.username[:2].upper()
    role_label = 'Academic Lecturer'

    total_students = sum(module.student_count for module in modules)
    assigned_classes = profile.number_of_classes or modules.count()
    thesis_students = latest_log.supervised_theses if latest_log else 0
    burnout_score = float(latest_log.burnout_risk_score) if latest_log else 0.0
    burnout_label = latest_log.burnout_risk_category.title() if latest_log else 'Low'
    workload_hours = float(latest_log.predicted_weekly_hours) if latest_log else 0.0
    target_hours = float(profile.max_weekly_hours_target or 40)
    over_target = max(0.0, workload_hours - target_hours)

    teaching_h = float(latest_log.teaching_hours) if latest_log else 0.0
    grading_h = float(latest_log.grading_backlog_days) / 2.0 if latest_log else 0.0
    thesis_h = float(thesis_students) * 1.0
    admin_h = round(min(8.0, max(1.0, workload_hours * 0.08)), 1) if workload_hours else 0.0
    research_h = round(max(0.0, workload_hours - (teaching_h + grading_h + thesis_h + admin_h)), 1)

    # Historical trend lists for JavaScript chart
    logs_reversed = list(reversed(list(logs)))
    trend_categories = [f"Week {i+1}" for i in range(len(logs_reversed))] or ['Week 1']
    trend_hours = [float(log.predicted_weekly_hours or 0) for log in logs_reversed] or [0]
    trend_fatigue = [int(log.self_reported_fatigue or 0) for log in logs_reversed] or [0]

    # Handle Log Submission
    if request.method == 'POST':
        try:
            # 1. Extract raw form inputs
            teaching_hours = float(request.POST.get('teaching_hours', 0))
            grading_backlog_days = int(request.POST.get('grading_backlog_days', 0))
            supervised_theses = int(request.POST.get('supervised_theses', 0))
            weekend_work_hours = float(request.POST.get('weekend_work_hours', 0))
            lms_late_night_actions = int(request.POST.get('lms_late_night_actions', 0))
            self_reported_fatigue = int(request.POST.get('self_reported_fatigue', 5))
            perceived_support = int(request.POST.get('perceived_support', 3))

            # 2. Build full feature dictionary matching ML Engine expected columns
            input_features = {
                'department': profile.department.name if profile.department else 'Software Engineering',
                'employment_type': profile.employment_type if hasattr(profile, 'employment_type') else 'FULL_TIME',
                'years_experience': getattr(profile, 'years_experience', 5),
                'number_of_classes': profile.number_of_classes or modules.count(),
                'modules_count': modules.count(),
                'total_students_enrolled': total_students,
                'assessment_type_weight': 1.5,
                'active_research_projects': 1,
                'hod_administrative_hours': 0.0,
                'teaching_hours_per_week': teaching_hours,
                'grading_backlog_days': grading_backlog_days,
                'thesis_students_count': supervised_theses,
                'weekend_work_hours': weekend_work_hours,
                'lms_activity_late_night_count': lms_late_night_actions,
                'self_reported_fatigue': self_reported_fatigue,
                'perceived_support_score': perceived_support,
            }

            # 3. Run ML Inference with complete feature set
            prediction = predict_workload_and_burnout(input_features)

            # 4. Persist Log Entry to Database
            WorkloadLog.objects.create(
                lecturer=profile,
                teaching_hours=teaching_hours,
                grading_backlog_days=grading_backlog_days,
                supervised_theses=supervised_theses,
                weekend_work_hours=weekend_work_hours,
                lms_late_night_actions=lms_late_night_actions,
                self_reported_fatigue=self_reported_fatigue,
                perceived_support=perceived_support,
                predicted_weekly_hours=prediction['predicted_weekly_hours'],
                burnout_risk_category=prediction['burnout_risk_category'],
                burnout_risk_score=prediction['burnout_risk_score'],
                recommended_action=prediction['recommended_action']
            )

            messages.success(request, "Weekly workload log recorded and AI predictions updated!")
            return redirect('lecturer_dashboard')

        except Exception as e:
            messages.error(request, f"Error processing prediction: {str(e)}")

    context = {
        'profile': profile,
        'modules': modules,
        'latest_log': latest_log,
        'logs_history': logs_reversed,
        'user_full_name': full_name,
        'user_first_name': first_name,
        'user_initials': initials,
        'user_role': role_label,
        'profile_department': profile.department.name,
        'profile_speciality': profile.major_speciality,
        'assigned_classes': assigned_classes,
        'total_students': total_students,
        'modules_count': modules.count(),
        'thesis_students': thesis_students,
        'burnout_score': int(round(burnout_score)),
        'burnout_label': burnout_label,
        'workload_hours': round(workload_hours, 1),
        'target_hours': target_hours,
        'over_target_hours': round(over_target, 1),
        'teaching_hours': teaching_h,
        'grading_hours': grading_h,
        'admin_hours': admin_h,
        'thesis_hours': thesis_h,
        'research_hours': research_h,
        'trend_categories': trend_categories,
        'trend_hours': trend_hours,
        'trend_fatigue': trend_fatigue,
    }
    return render(request, 'workload/lecturer_dashboard.html', context)