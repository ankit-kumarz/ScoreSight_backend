from django.db import models
from django.contrib.auth.models import User


class UserProfile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    favorite_team = models.CharField(max_length=100)
    predictions_count = models.IntegerField(default=0)
    correct_predictions = models.IntegerField(default=0)

    def __str__(self):
        return f"{self.user.username}'s profile"

    @property
    def accuracy(self):
        if self.predictions_count == 0:
            return 0
        return (self.correct_predictions / self.predictions_count) * 100


class PredictionHistory(models.Model):
    timestamp = models.DateTimeField(auto_now_add=True)
    input_data = models.JSONField()
    outcome = models.CharField(max_length=4)
    probabilities = models.JSONField()
    goal_diff = models.FloatField(null=True)
    suggested_score = models.JSONField()

    def __str__(self):
        return f"Prediction {self.timestamp} - {self.outcome}"
