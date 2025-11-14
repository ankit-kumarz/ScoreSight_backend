from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework import status
from rest_framework.permissions import IsAuthenticated, AllowAny
from django.shortcuts import get_object_or_404
from django.contrib.auth.models import User
from django.contrib.auth import authenticate, login
from .serializers import (
    HealthSerializer,
    TeamListSerializer,
    PredictRequestSerializer,
    PredictResponseSerializer,
    UserSerializer,
)
from .inference import get_inferencer
from .models import PredictionHistory, UserProfile
import pandas as pd


class HealthView(APIView):
    def get(self, request):
        data = {'status': 'ok', 'model': 'sklearn', 'version': 'v2'}
        return Response(HealthSerializer(data).data)


class DebugInputView(APIView):
    """Return expected feature vector format and a sample vector for quick debugging."""

    def get(self, request):
        inf = get_inferencer()
        features = inf.features() if hasattr(inf, 'features') else []
        sample = [0 for _ in features]
        return Response({'features': features, 'sample_vector': sample})


class TeamsView(APIView):
    def get(self, request):
        inf = get_inferencer()
        data = {'teams': inf.teams()}
        return Response(TeamListSerializer(data).data)


class UserStatsView(APIView):
    permission_classes = [IsAuthenticated]
    
    def get(self, request):
        profile = get_object_or_404(UserProfile, user=request.user)
        return Response({
            'total_predictions': profile.predictions_count,
            'accuracy': profile.accuracy,
            'favorite_team': profile.favorite_team
        })


class SignupView(APIView):
    permission_classes = [AllowAny]
    
    def post(self, request):
        name = request.data.get('name')
        email = request.data.get('email')
        password = request.data.get('password')
        
        if not name or not email or not password:
            return Response(
                {'status': 'error', 'message': 'Name, email, and password are required.'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # Check if user already exists
        if User.objects.filter(email=email).exists():
            return Response(
                {'status': 'error', 'message': 'Email already registered.'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        try:
            # Create user with email as username
            user = User.objects.create_user(
                username=email,
                email=email,
                password=password,
                first_name=name
            )
            
            # Create user profile
            UserProfile.objects.create(
                user=user,
                favorite_team='',
                predictions_count=0,
                correct_predictions=0
            )
            
            return Response({
                'status': 'success',
                'name': name,
                'email': email
            }, status=status.HTTP_201_CREATED)
        except Exception as e:
            return Response(
                {'status': 'error', 'message': str(e)},
                status=status.HTTP_400_BAD_REQUEST
            )


class LoginView(APIView):
    permission_classes = [AllowAny]
    
    def post(self, request):
        email = request.data.get('email')
        password = request.data.get('password')
        
        if not email or not password:
            return Response(
                {'status': 'error', 'message': 'Email and password are required.'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        try:
            # Find user by email
            user = User.objects.get(email=email)
            
            # Authenticate with username (which is same as email)
            user = authenticate(username=user.username, password=password)
            
            if user:
                login(request, user)
                name = user.first_name or user.username
                return Response({
                    'status': 'success',
                    'name': name,
                    'email': user.email
                }, status=status.HTTP_200_OK)
            else:
                return Response(
                    {'status': 'error', 'message': 'Invalid email or password.'},
                    status=status.HTTP_401_UNAUTHORIZED
                )
        except User.DoesNotExist:
            return Response(
                {'status': 'error', 'message': 'Invalid email or password.'},
                status=status.HTTP_401_UNAUTHORIZED
            )
        except Exception as e:
            return Response(
                {'status': 'error', 'message': str(e)},
                status=status.HTTP_400_BAD_REQUEST
            )

class PredictV2View(APIView):
    permission_classes = [AllowAny]  # Changed to allow anonymous predictions for demo
    
    def post(self, request):
        # Handle both frontend simple request and complex stats request
        home_team = request.data.get('home_team') or request.data.get('HomeTeam')
        away_team = request.data.get('away_team') or request.data.get('AwayTeam')
        match_date = request.data.get('match_date')
        
        if not home_team or not away_team:
            return Response(
                {'error': 'home_team and away_team are required.'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        inf = get_inferencer()
        
        # Create a simple prediction request with just team names
        # The inferencer will use default values for stats
        match_data = {
            'HomeTeam': home_team,
            'AwayTeam': away_team,
            'HTHG': request.data.get('HTHG', 0),
            'HTAG': request.data.get('HTAG', 0),
            'HS': request.data.get('HS', 0),
            'AS': request.data.get('AS', 0),
            'HST': request.data.get('HST', 0),
            'AST': request.data.get('AST', 0),
            'HC': request.data.get('HC', 0),
            'AC': request.data.get('AC', 0),
            'HF': request.data.get('HF', 0),
            'AF': request.data.get('AF', 0),
            'HY': request.data.get('HY', 0),
            'AY': request.data.get('AY', 0),
            'HR': request.data.get('HR', 0),
            'AR': request.data.get('AR', 0)
        }
        
        try:
            res = inf.predict_single(match_data)
            
            # Transform response to match frontend expectations
            # Extract probabilities and create structured response
            probs_raw = res.get('probabilities', [])
            labels = inf.class_labels()  # ['H', 'D', 'A']
            
            # Create probability dict with clear labels
            prob_dict = {}
            for item in probs_raw:
                label = item['label']
                prob = item['prob']
                if label == 'H':
                    prob_dict['home_win'] = prob
                elif label == 'D':
                    prob_dict['draw'] = prob
                elif label == 'A':
                    prob_dict['away_win'] = prob
            
            # Ensure all keys exist (fallback to 0)
            prob_dict.setdefault('home_win', 0.0)
            prob_dict.setdefault('draw', 0.0)
            prob_dict.setdefault('away_win', 0.0)
            
            # Determine outcome text
            outcome_code = res.get('outcome', 'D')
            if outcome_code == 'H':
                outcome_text = 'Home Win'
            elif outcome_code == 'A':
                outcome_text = 'Away Win'
            else:
                outcome_text = 'Draw'
            
            # Calculate points based on outcome
            if outcome_code == 'H':
                home_points = 3
                away_points = 0
            elif outcome_code == 'A':
                home_points = 0
                away_points = 3
            else:
                home_points = 1
                away_points = 1
            
            response_data = {
                'outcome': outcome_text,
                'probabilities': prob_dict,
                'home_goals': res.get('suggested_score', {}).get('home', 1),
                'away_goals': res.get('suggested_score', {}).get('away', 1),
                'home_points': home_points,
                'away_points': away_points,
                'goal_difference': round(res.get('goal_diff', 0), 2)
            }
            
            # Save prediction history if user is authenticated
            if request.user.is_authenticated:
                try:
                    profile = UserProfile.objects.get(user=request.user)
                    profile.predictions_count += 1
                    profile.save()
                    
                    PredictionHistory.objects.create(
                        input_data={'home_team': home_team, 'away_team': away_team, 'match_date': match_date},
                        outcome=response_data.get('outcome', ''),
                        probabilities=response_data.get('probabilities', {}),
                        goal_diff=response_data.get('goal_difference'),
                        suggested_score={'home': response_data.get('home_goals'), 'away': response_data.get('away_goals')},
                    )
                except Exception:
                    pass  # DB optional - don't fail prediction
            
            return Response(response_data, status=status.HTTP_200_OK)
        except Exception as e:
            return Response(
                {'error': str(e)},
                status=status.HTTP_400_BAD_REQUEST
            )


class SimulateView(APIView):
    parser_classes = (MultiPartParser, FormParser)

    def post(self, request):
        # expect uploaded CSV file under 'file'
        csv_file = request.FILES.get('file')
        if csv_file is None:
            return Response({'detail': 'file is required'}, status=status.HTTP_400_BAD_REQUEST)

        try:
            df = pd.read_csv(csv_file, parse_dates=['Date'], dayfirst=True)
        except Exception:
            try:
                df = pd.read_csv(csv_file)
            except Exception:
                return Response({'detail': 'unable to parse CSV'}, status=status.HTTP_400_BAD_REQUEST)

        inf = get_inferencer()

        # standings dict
        table = {}

        def ensure_team(t):
            if t not in table:
                table[t] = {'team': t, 'played': 0, 'wins': 0, 'draws': 0, 'losses': 0, 'points': 0, 'gf': 0, 'ga': 0}

        for _, row in df.iterrows():
            home = row.get('HomeTeam')
            away = row.get('AwayTeam')
            if pd.isna(home) or pd.isna(away):
                continue
            payload = {'HomeTeam': home, 'AwayTeam': away}
            pr = inf.predict_single(payload)
            score = pr.get('suggested_score', {'home': 0, 'away': 0})
            hg = int(score.get('home', 0))
            ag = int(score.get('away', 0))

            ensure_team(home); ensure_team(away)
            table[home]['played'] += 1
            table[away]['played'] += 1
            table[home]['gf'] += hg
            table[home]['ga'] += ag
            table[away]['gf'] += ag
            table[away]['ga'] += hg

            if hg > ag:
                table[home]['wins'] += 1
                table[away]['losses'] += 1
                table[home]['points'] += 3
            elif hg == ag:
                table[home]['draws'] += 1
                table[away]['draws'] += 1
                table[home]['points'] += 1
                table[away]['points'] += 1
            else:
                table[away]['wins'] += 1
                table[home]['losses'] += 1
                table[away]['points'] += 3

        standings = sorted(table.values(), key=lambda r: (r['points'], r['gf'] - r['ga']), reverse=True)
        return Response({'standings': standings})
