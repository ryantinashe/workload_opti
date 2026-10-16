from django.contrib.auth import get_user_model
from django.test import TestCase
from django.urls import reverse

from users.models import Department, LecturerProfile
from workload.models import WorkloadLog

User = get_user_model()


class HodDashboardDataTests(TestCase):
    def setUp(self):
        self.hod_user = User.objects.create_user(
            username='hod_user',
            email='hod@example.com',
            password='securepass123',
            role=User.Role.HOD,
        )
        self.department = Department.objects.create(name='Computing', code='CMP', hod=self.hod_user)
        LecturerProfile.objects.create(
            user=self.hod_user,
            department=self.department,
            major_speciality='Academic Leadership',
            related_specialities='Strategy, Research',
            years_experience=10,
            number_of_classes=2,
            max_weekly_hours_target=18,
        )

        self.lecturer_user = User.objects.create_user(
            username='lecturer_one',
            email='lecturer@example.com',
            password='securepass123',
            role=User.Role.LECTURER,
        )
        self.lecturer_profile = LecturerProfile.objects.create(
            user=self.lecturer_user,
            department=self.department,
            major_speciality='Software Engineering',
            related_specialities='Python, Django',
            years_experience=4,
            number_of_classes=2,
            max_weekly_hours_target=18,
        )
        WorkloadLog.objects.create(
            lecturer=self.lecturer_profile,
            teaching_hours=8.5,
            grading_backlog_days=4,
            supervised_theses=2,
            predicted_weekly_hours=14.5,
            burnout_risk_score=58.0,
            burnout_risk_category='HIGH',
            recommended_action='Reduce teaching load',
        )

    def test_hod_dashboard_uses_db_lecturer_data_not_hardcoded_values(self):
        self.client.force_login(self.hod_user)
        response = self.client.get(reverse('hod_dashboard'))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'initialDashboardData')
        self.assertContains(response, 'lecturer_one')
        self.assertContains(response, 'Software Engineering')
        self.assertNotContains(response, 'Dr. Emmanuel N.')

    def test_critical_lecturers_show_extra_hours_only_when_above_40h_threshold(self):
        latest_log = self.lecturer_profile.workload_logs.order_by('-logged_at').first()
        latest_log.teaching_hours = 28.0
        latest_log.grading_backlog_days = 10
        latest_log.supervised_theses = 3
        latest_log.predicted_weekly_hours = 48.0
        latest_log.burnout_risk_score = 88.0
        latest_log.burnout_risk_category = 'CRITICAL'
        latest_log.save()

        self.client.force_login(self.hod_user)
        response = self.client.get(reverse('hod_dashboard'))

        self.assertEqual(response.status_code, 200)
        self.assertGreater(
            response.context['initial_dashboard_data']['dashboardStats']['extraTeachingHours'],
            0,
        )

    def test_apply_ai_rebalancing_returns_clear_audit_and_updates_workload(self):
        WorkloadLog.objects.create(
            lecturer=self.lecturer_profile,
            teaching_hours=20.0,
            grading_backlog_days=12,
            supervised_theses=8,
            predicted_weekly_hours=24.0,
            burnout_risk_score=90.0,
            burnout_risk_category='CRITICAL',
            recommended_action='Critical overload',
        )

        self.client.force_login(self.hod_user)
        response = self.client.post(reverse('apply_ai_rebalancing'))

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn('AI rebalancing applied', data['message'])
        self.assertIn('safer workload', data['message'].lower())
        self.assertIn('lecturerBreakdown', data['details'])
        self.assertGreater(len(data['details']['lecturerBreakdown']), 0)

        latest_log = self.lecturer_profile.workload_logs.order_by('-logged_at').first()
        self.assertLessEqual(latest_log.predicted_weekly_hours, 18.0)
        self.assertIn(latest_log.burnout_risk_category, ['LOW', 'MODERATE'])
