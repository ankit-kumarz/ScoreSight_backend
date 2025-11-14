from rest_framework import serializers
from django.contrib.auth.models import User


class UserSerializer(serializers.ModelSerializer):
    password = serializers.CharField(write_only=True)
    favorite_team = serializers.CharField(write_only=True, required=False)

    class Meta:
        model = User
        fields = ('id', 'username', 'email', 'password', 'favorite_team')

    def create(self, validated_data):
        favorite_team = validated_data.pop('favorite_team', '')
        user = User.objects.create_user(
            username=validated_data['username'],
            email=validated_data.get('email', ''),
            password=validated_data['password']
        )
        return user


class HealthSerializer(serializers.Serializer):
    status = serializers.CharField()
    model = serializers.CharField()
    version = serializers.CharField()


class TeamListSerializer(serializers.Serializer):
    teams = serializers.ListField(child=serializers.CharField())


class PredictRequestSerializer(serializers.Serializer):
    HomeTeam = serializers.CharField()
    AwayTeam = serializers.CharField()
    # Accept numeric match stats - flexible keys
    HTHG = serializers.IntegerField(required=False, allow_null=True)
    HTAG = serializers.IntegerField(required=False, allow_null=True)
    HS = serializers.IntegerField(required=False, allow_null=True)
    AS = serializers.IntegerField(required=False, allow_null=True)
    HST = serializers.IntegerField(required=False, allow_null=True)
    AST = serializers.IntegerField(required=False, allow_null=True)
    HC = serializers.IntegerField(required=False, allow_null=True)
    AC = serializers.IntegerField(required=False, allow_null=True)
    HF = serializers.IntegerField(required=False, allow_null=True)
    AF = serializers.IntegerField(required=False, allow_null=True)
    HY = serializers.IntegerField(required=False, allow_null=True)
    AY = serializers.IntegerField(required=False, allow_null=True)
    HR = serializers.IntegerField(required=False, allow_null=True)
    AR = serializers.IntegerField(required=False, allow_null=True)


class ProbabilitySerializer(serializers.Serializer):
    label = serializers.CharField()
    prob = serializers.FloatField()


class PredictResponseSerializer(serializers.Serializer):
    outcome = serializers.CharField()
    probabilities = ProbabilitySerializer(many=True)
    goal_diff = serializers.FloatField()
    suggested_score = serializers.DictField()
