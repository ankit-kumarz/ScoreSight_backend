from django.test import TestCase
from django.urls import reverse
from rest_framework.test import APIClient


class ApiSmokeTests(TestCase):
    def setUp(self):
        self.client = APIClient()

    def test_health(self):
        resp = self.client.get('/api/health')
        self.assertEqual(resp.status_code, 200)
        self.assertIn('status', resp.json())

    def test_predict_missing(self):
        resp = self.client.post('/api/predict_v2', data={})
        self.assertEqual(resp.status_code, 400)
