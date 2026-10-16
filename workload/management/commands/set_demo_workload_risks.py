from django.core.management.base import BaseCommand
from django.db import transaction

from users.models import LecturerProfile
from workload.models import WorkloadLog


RISK_PROFILES = (
    {
        'category': 'LOW',
        'score': 24.0,
        'teaching_hours': 12.0,
        'grading_backlog_days': 1,
        'supervised_theses': 1,
        'weekend_work_hours': 0.0,
        'lms_late_night_actions': 2,
        'fatigue': 3,
        'support': 5,
        'predicted_hours': 18.0,
        'action': 'Optimal capacity: lecturer can accept limited additional academic or research duties.',
    },
    {
        'category': 'MODERATE',
        'score': 46.0,
        'teaching_hours': 18.0,
        'grading_backlog_days': 4,
        'supervised_theses': 3,
        'weekend_work_hours': 2.0,
        'lms_late_night_actions': 6,
        'fatigue': 5,
        'support': 3,
        'predicted_hours': 28.0,
        'action': 'Manageable strain: monitor grading backlog and upcoming assessment periods.',
    },
    {
        'category': 'HIGH',
        'score': 68.0,
        'teaching_hours': 24.0,
        'grading_backlog_days': 8,
        'supervised_theses': 5,
        'weekend_work_hours': 5.0,
        'lms_late_night_actions': 12,
        'fatigue': 7,
        'support': 2,
        'predicted_hours': 38.0,
        'action': 'Elevated strain: reallocate grading support or thesis supervision before workload increases.',
    },
    {
        'category': 'CRITICAL',
        'score': 90.0,
        'teaching_hours': 28.0,
        'grading_backlog_days': 12,
        'supervised_theses': 8,
        'weekend_work_hours': 8.0,
        'lms_late_night_actions': 18,
        'fatigue': 10,
        'support': 1,
        'predicted_hours': 48.0,
        'action': 'Critical overload: immediate HOD intervention and workload redistribution required.',
    },
)


class Command(BaseCommand):
    help = 'Creates varied low, moderate, high, and critical workload profiles for dashboard testing.'

    @transaction.atomic
    def handle(self, *args, **options):
        lecturers = list(LecturerProfile.objects.select_related('user').order_by('id'))
        if not lecturers:
            self.stdout.write(self.style.ERROR('No lecturer profiles found. Run seed_db first.'))
            return

        updated_logs = 0
        profile_counts = {profile['category']: 0 for profile in RISK_PROFILES}

        for index, lecturer in enumerate(lecturers):
            profile = RISK_PROFILES[index % len(RISK_PROFILES)]
            profile_counts[profile['category']] += 1

            lecturer.number_of_classes = 1 if profile['category'] == 'LOW' else 2
            lecturer.max_weekly_hours_target = 18
            lecturer.save(update_fields=['number_of_classes', 'max_weekly_hours_target', 'updated_at'])

            workload_logs = lecturer.workload_logs.all()
            for log in workload_logs:
                log.teaching_hours = profile['teaching_hours']
                log.grading_backlog_days = profile['grading_backlog_days']
                log.supervised_theses = profile['supervised_theses']
                log.weekend_work_hours = profile['weekend_work_hours']
                log.lms_late_night_actions = profile['lms_late_night_actions']
                log.self_reported_fatigue = profile['fatigue']
                log.perceived_support = profile['support']
                log.predicted_weekly_hours = profile['predicted_hours']
                log.burnout_risk_category = profile['category']
                log.burnout_risk_score = profile['score']
                log.recommended_action = profile['action']
                log.save(update_fields=[
                    'teaching_hours',
                    'grading_backlog_days',
                    'supervised_theses',
                    'weekend_work_hours',
                    'lms_late_night_actions',
                    'self_reported_fatigue',
                    'perceived_support',
                    'predicted_weekly_hours',
                    'burnout_risk_category',
                    'burnout_risk_score',
                    'recommended_action',
                ])
                updated_logs += 1

        self.stdout.write(self.style.SUCCESS(f'Updated {updated_logs} workload logs across {len(lecturers)} lecturers.'))
        for category, count in profile_counts.items():
            self.stdout.write(f'{category}: {count} lecturers')
