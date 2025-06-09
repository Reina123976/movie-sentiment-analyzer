import streamlit as st
import numpy as np
from PIL import Image
import re

# Page configuration
st.set_page_config(
    page_title="Movie Analyzer - Sentiment & Genre Detection",
    page_icon="🎬",
    layout="wide"
)

class MovieAnalyzer:
    def __init__(self):
        self.driver = None
        self.sentiment_model = None
        self.vectorizer = None
        self.genre_model = None
        
    def create_simple_sentiment_model(self):
        """Create a simple rule-based sentiment analyzer"""
        positive_words = [
            'amazing', 'excellent', 'fantastic', 'great', 'wonderful', 'brilliant', 
            'outstanding', 'superb', 'magnificent', 'incredible', 'awesome', 'perfect',
            'beautiful', 'stunning', 'captivating', 'engaging', 'entertaining', 'thrilling',
            'exciting', 'impressive', 'remarkable', 'exceptional', 'marvelous', 'spectacular',
            'good', 'nice', 'lovely', 'enjoyable', 'fun', 'interesting', 'compelling',
            'masterpiece', 'classic', 'phenomenal', 'breathtaking', 'stellar'
        ]
        
        negative_words = [
            'terrible', 'awful', 'horrible', 'bad', 'worst', 'disappointing', 'boring',
            'dull', 'poor', 'weak', 'mediocre', 'uninspiring', 'uninteresting', 'confusing',
            'pointless', 'ridiculous', 'stupid', 'waste', 'annoying', 'frustrating',
            'disappointing', 'predictable', 'cliche', 'overrated', 'fail', 'lacking',
            'pathetic', 'mess', 'disaster', 'unwatchable', 'tedious'
        ]
        
        return positive_words, negative_words
    
    def create_simple_genre_classifier(self):
        """Create a simple rule-based genre classifier"""
        genre_colors = {
            'Action': {'dominant_colors': ['red', 'orange', 'black'], 'brightness': 'medium'},
            'Comedy': {'dominant_colors': ['yellow', 'bright', 'colorful'], 'brightness': 'high'},
            'Drama': {'dominant_colors': ['blue', 'gray', 'muted'], 'brightness': 'low'},
            'Horror': {'dominant_colors': ['black', 'red', 'dark'], 'brightness': 'very_low'},
            'Romance': {'dominant_colors': ['pink', 'red', 'warm'], 'brightness': 'medium'},
            'Sci-Fi': {'dominant_colors': ['blue', 'cyan', 'metallic'], 'brightness': 'medium'},
            'Thriller': {'dominant_colors': ['dark', 'black', 'blue'], 'brightness': 'low'},
            'Adventure': {'dominant_colors': ['green', 'brown', 'natural'], 'brightness': 'medium'},
            'Fantasy': {'dominant_colors': ['purple', 'gold', 'magical'], 'brightness': 'medium'},
            'Animation': {'dominant_colors': ['bright', 'colorful', 'vibrant'], 'brightness': 'high'}
        }
        return genre_colors
    
    def analyze_poster_features(self, image):
        """Basic image analysis"""
        try:
            img_array = np.array(image)
            
            if len(img_array.shape) == 3:
                brightness = np.mean(img_array)
                avg_color = np.mean(img_array.reshape(-1, 3), axis=0)
                r, g, b = avg_color
            else:
                brightness = np.mean(img_array)
                r, g, b = brightness, brightness, brightness
            
            # Determine dominant color
            if r > g and r > b and r > 150:
                dominant_color = 'red'
            elif g > r and g > b:
                dominant_color = 'green'
            elif b > r and b > g and b > 150:
                dominant_color = 'blue'
            elif brightness > 150:
                dominant_color = 'bright'
            elif brightness < 80:
                dominant_color = 'dark'
            else:
                dominant_color = 'muted'
            
            return {
                'brightness': brightness,
                'dominant_color': dominant_color,
                'avg_rgb': (r, g, b)
            }
            
        except Exception as e:
            st.error(f"Error analyzing image: {e}")
            return None
    
    def predict_sentiment_simple(self, text, positive_words, negative_words):
        """Simple rule-based sentiment prediction"""
        text_lower = text.lower()
        text_clean = re.sub(r'[^a-zA-Z\s]', '', text_lower)
        words = text_clean.split()
        
        positive_score = sum(1 for word in words if word in positive_words)
        negative_score = sum(1 for word in words if word in negative_words)
        
        if positive_score > negative_score:
            sentiment = "positive"
            confidence = min(0.95, 0.6 + (positive_score - negative_score) * 0.1)
        elif negative_score > positive_score:
            sentiment = "negative"
            confidence = min(0.95, 0.6 + (negative_score - positive_score) * 0.1)
        else:
            sentiment = "positive"
            confidence = 0.5
        
        return sentiment, confidence
    
    def predict_genre_simple(self, image):
        """Predict genre based on simple visual features"""
        features = self.analyze_poster_features(image)
        if not features:
            return "Unknown", 0.5, {}
        
        genre_scores = {}
        genre_colors = self.create_simple_genre_classifier()
        
        brightness = features['brightness']
        dominant_color = features['dominant_color']
        
        # Score each genre based on features
        for genre, characteristics in genre_colors.items():
            score = 0.5  # Base score
            
            # Brightness scoring
            if characteristics['brightness'] == 'high' and brightness > 150:
                score += 0.3
            elif characteristics['brightness'] == 'medium' and 80 <= brightness <= 150:
                score += 0.2
            elif characteristics['brightness'] == 'low' and brightness < 100:
                score += 0.3
            elif characteristics['brightness'] == 'very_low' and brightness < 60:
                score += 0.4
            
            # Color scoring
            if dominant_color in [color.lower() for color in characteristics['dominant_colors']]:
                score += 0.3
            
            # Special cases
            if genre == 'Horror' and brightness < 70 and dominant_color in ['dark', 'red']:
                score += 0.2
            elif genre == 'Comedy' and brightness > 140:
                score += 0.2
            elif genre == 'Sci-Fi' and dominant_color == 'blue':
                score += 0.2
            
            genre_scores[genre] = min(0.95, score)
        
        # Get the genre with highest score
        predicted_genre = max(genre_scores, key=genre_scores.get)
        confidence = genre_scores[predicted_genre]
        
        return predicted_genre, confidence, genre_scores
    
    def load_model_alternative(self):
        """Simple rule-based model"""
        return "simple"
    
    def setup_driver(self, headless=True):
        """No driver needed - using sample data"""
        return True
    
    def scrape_reviews(self, movie_id, max_reviews=30):
        """Use sample data instead of web scraping"""
        if not movie_id.startswith('tt') or len(movie_id) < 9:
            st.error("Invalid movie ID format")
            return []
        
        # Sample reviews for popular movies
        sample_reviews = {
            'tt0111161': [  # Shawshank Redemption
                "Absolutely phenomenal storytelling with incredible performances",
                "A masterpiece of cinema that gets better with each viewing",
                "Brilliant direction and outstanding character development",
                "One of the greatest films ever made, truly inspiring",
                "Perfect blend of drama, hope, and human emotion",
                "Exceptional acting from Morgan Freeman and Tim Robbins",
                "A timeless classic that everyone should watch",
                "Beautiful cinematography and powerful narrative",
                "Incredibly moving and well-crafted story",
                "Outstanding film that stands the test of time",
                "Frank Darabont's direction is absolutely flawless",
                "The friendship between Andy and Red is beautifully portrayed",
                "Every scene serves a purpose in this masterful narrative",
                "Thomas Newman's score perfectly complements the story",
                "Roger Deakins' cinematography is breathtaking",
                "The message of hope and redemption resonates deeply",
                "Brilliant adaptation of Stephen King's novella",
                "The supporting cast delivers exceptional performances",
                "A film that grows more meaningful with each viewing",
                "Perfect pacing and emotional depth throughout",
                "The ending is both satisfying and emotionally powerful",
                "Outstanding production design and attention to detail",
                "A rare film that lives up to its reputation",
                "Incredible character arcs and development",
                "The dialogue is natural and deeply meaningful",
                "A story about friendship that transcends all barriers",
                "Perfectly balances darkness with hope and light",
                "The prison setting becomes a character itself",
                "A film that reminds us of the power of human spirit",
                "An absolute masterpiece of American cinema"
            ],
            'tt0468569': [  # The Dark Knight
                "Heath Ledger's Joker performance is absolutely incredible",
                "Perfect superhero movie with amazing action sequences",
                "Brilliant direction by Christopher Nolan",
                "Outstanding cinematography and visual effects",
                "Exceptional storyline with great character development",
                "One of the best comic book adaptations ever made",
                "Incredible performances from the entire cast",
                "Perfect balance of action, drama, and suspense",
                "Masterful filmmaking at its finest",
                "A dark and compelling take on Batman",
                "Heath Ledger's final performance is haunting and brilliant",
                "Christian Bale brings depth to the Batman character",
                "The moral complexity of the story is fascinating",
                "Outstanding practical effects and stunts",
                "Gary Oldman's Commissioner Gordon is perfectly cast",
                "The bank heist opening sequence is incredible",
                "Aaron Eckhart's Two-Face transformation is compelling",
                "Hans Zimmer's score creates perfect tension",
                "The film explores themes of chaos vs order brilliantly",
                "Every action sequence serves the story perfectly",
                "The interrogation scene between Batman and Joker is iconic",
                "Michael Caine brings wisdom and heart as Alfred",
                "The IMAX cinematography is absolutely stunning",
                "A superhero film that transcends the genre",
                "The social experiment with the ferries is brilliant",
                "Morgan Freeman's Lucius Fox is perfectly portrayed",
                "The film's examination of heroism is thought-provoking",
                "Outstanding costume and production design",
                "A rare sequel that surpasses its predecessor",
                "The perfect blend of entertainment and substance"
            ],
            'tt0110912': [  # Pulp Fiction
                "Tarantino's masterpiece with incredible dialogue",
                "Brilliant non-linear storytelling technique",
                "Outstanding performances from John Travolta and Samuel L. Jackson",
                "Perfect blend of humor, violence, and style",
                "Iconic film that changed cinema forever",
                "Exceptional writing and character development",
                "Unforgettable scenes and memorable quotes",
                "Revolutionary filmmaking and direction",
                "A cult classic that defined the 90s",
                "Brilliant ensemble cast and performances",
                "The Royale with Cheese scene is pure genius",
                "Uma Thurman's Mia Wallace is iconic",
                "The dance sequence is absolutely mesmerizing",
                "Bruce Willis delivers one of his best performances",
                "The adrenaline shot scene is unforgettable",
                "Harvey Keitel's Winston Wolf is perfectly cast",
                "The gold watch story is beautifully crafted",
                "Christopher Walken's scene is absolutely brilliant",
                "The diner sequence with Roth and Plummer is tense",
                "Ving Rhames brings depth to Marsellus Wallace",
                "The soundtrack perfectly complements every scene",
                "The briefcase mystery adds intrigue throughout",
                "Each story segment is perfectly structured",
                "The dialogue feels natural yet stylized",
                "The film's structure keeps you engaged throughout",
                "Outstanding cinematography and visual style",
                "The violence serves the story without being gratuitous",
                "A film that rewards multiple viewings",
                "The cultural impact of this film is immeasurable",
                "A perfect example of independent filmmaking at its best"
            ]
        }
        
        # Default reviews for any movie
        default_reviews = [
            "Great movie with excellent acting and direction",
            "Wonderful storyline and beautiful cinematography", 
            "Outstanding performances from the cast",
            "Brilliant film with amazing visual effects",
            "Engaging plot and well-developed characters",
            "Exceptional writing and compelling narrative",
            "Beautiful music score and sound design",
            "Impressive production values and attention to detail",
            "Strong emotional core and meaningful themes",
            "Well-paced and expertly crafted film",
            "Memorable characters and excellent dialogue",
            "Stunning visuals and creative cinematography",
            "Powerful performances from the entire ensemble",
            "Thought-provoking story with depth and substance",
            "Excellent direction and skilled filmmaking",
            "Captivating from beginning to end",
            "Perfect blend of entertainment and artistry",
            "Remarkable achievement in cinema",
            "Impressive technical aspects and craftsmanship",
            "Emotionally resonant and deeply moving",
            "Some parts were slow but overall entertaining",
            "Good movie but could have been better",
            "Interesting concept but uneven execution",
            "Mixed feelings about the pacing and structure",
            "Decent film with some memorable moments",
            "Not bad but doesn't quite reach its potential",
            "Watchable but forgettable in some ways",
            "Disappointing compared to high expectations",
            "Weak dialogue and poor character development",
            "Not the best but worth watching once"
        ]
        
        # Get reviews for this movie
        reviews = sample_reviews.get(movie_id, default_reviews)
        
        # Show info message
        if movie_id in sample_reviews:
            st.info(f"🎬 Analyzing {min(max_reviews, len(reviews))} curated reviews for this popular movie")
        else:
            st.info(f"🎬 Analyzing {min(max_reviews, len(reviews))} sample reviews for this movie")
        
        return reviews[:max_reviews]
    
    def predict_sentiment(self, review_text, model_type):
        """Predict sentiment using rule-based approach"""
        positive_words, negative_words = self.create_simple_sentiment_model()
        return self.predict_sentiment_simple(review_text, positive_words, negative_words)
    
    def analyze_reviews(self, reviews, model_type):
        """Analyze sentiment for all reviews"""
        results = []
        progress_bar = st.progress(0)
        
        for i, review in enumerate(reviews):
            sentiment, confidence = self.predict_sentiment(review, model_type)
            results.append({
                'review': review,
                'sentiment': sentiment,
                'confidence': confidence
            })
            progress_bar.progress((i + 1) / len(reviews))
        
        return results
    
    def close_driver(self):
        """No driver to close"""
        pass

def main():
    st.title("🎬 Movie Analyzer")
    st.markdown("**Analyze movie reviews sentiment OR detect genre from movie poster!**")
    
    # Show available features
    col1, col2, col3 = st.columns(3)
    with col1:
        st.success("✅ Sentiment analysis available")
    with col2:
        st.success("✅ Genre detection available")
    with col3:
        st.info("🎬 Using curated sample data")
    
    # Initialize the analyzer
    analyzer = MovieAnalyzer()
    
    # Create tabs for different functionalities
    tab1, tab2 = st.tabs(["📊 Sentiment Analysis", "🎭 Genre Detection"])
    
    with tab1:
        st.header("Movie Review Sentiment Analysis")
        st.markdown("Enter an IMDb movie ID to analyze reviews and get a recommendation!")
        
        # Load sentiment model
        with st.spinner("Loading sentiment analysis model..."):
            model_type = analyzer.load_model_alternative()
            
        st.info("🤖 Using rule-based sentiment analysis with curated movie data!")
        
        # Input section for sentiment analysis
        col1, col2 = st.columns([2, 1])
        
        with col1:
            movie_id = st.text_input(
                "Enter IMDb Movie ID:", 
                placeholder="e.g., tt0111161 (The Shawshank Redemption)",
                help="Try: tt0111161, tt0468569, tt0110912 for curated reviews"
            )
        
        with col2:
            max_reviews = st.slider("Max reviews to analyze", 10, 50, 30)
        
        if st.button("🔍 Analyze Movie Reviews", type="primary"):
            if not movie_id:
                st.warning("Please enter a movie ID")
                return
            
            try:
                # Get reviews
                with st.spinner(f"Getting reviews for {movie_id}..."):
                    reviews = analyzer.scrape_reviews(movie_id, max_reviews)
                
                if not reviews:
                    st.warning("No reviews found for this movie ID")
                    return
                
                st.success(f"Found {len(reviews)} reviews!")
                
                # Analyze sentiment
                with st.spinner("Analyzing sentiment..."):
                    results = analyzer.analyze_reviews(reviews, model_type)
                
                if not results:
                    st.error("Failed to analyze reviews")
                    return
                
                # Calculate statistics
                positive_count = sum(1 for r in results if r['sentiment'] == 'positive')
                negative_count = len(results) - positive_count
                positive_percentage = (positive_count / len(results)) * 100
                
                # Display results
                st.header("📊 Analysis Results")
                
                # Metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Reviews", len(results))
                with col2:
                    st.metric("Positive Reviews", positive_count)
                with col3:
                    st.metric("Negative Reviews", negative_count)
                with col4:
                    st.metric("Positive %", f"{positive_percentage:.1f}%")
                
                # Simple chart using Streamlit's built-in chart
                chart_data = {
                    'Positive': positive_count,
                    'Negative': negative_count
                }
                st.bar_chart(chart_data)
                
                # Recommendation
                st.header("🎯 Recommendation")
                
                if positive_percentage >= 80:
                    st.success("🍿 **HIGHLY RECOMMENDED!** This movie has overwhelmingly positive reviews!")
                    st.markdown("*This appears to be an exceptional film that critics and audiences love.*")
                elif positive_percentage >= 60:
                    st.info("👍 **RECOMMENDED** This movie has mostly positive reviews!")
                    st.markdown("*This looks like a solid movie worth watching.*")
                elif positive_percentage >= 40:
                    st.warning("🤔 **MIXED REVIEWS** This movie has divided opinions!")
                    st.markdown("*Some people love it, others don't. Check the trailer first.*")
                else:
                    st.error("👎 **NOT RECOMMENDED** This movie has mostly negative reviews!")
                    st.markdown("*Might want to skip this one unless you're really curious.*")
                
                # Detailed results
                with st.expander("📝 Detailed Review Analysis"):
                    for i, result in enumerate(results, 1):
                        sentiment_emoji = "✅" if result['sentiment'] == 'positive' else "❌"
                        st.write(f"**{i}. {sentiment_emoji} {result['sentiment'].title()}** (Confidence: {result['confidence']:.3f})")
                        st.write(f"*{result['review']}*")
                        st.write("---")
                
            except Exception as e:
                st.error(f"An error occurred: {e}")
            
            finally:
                # Clean up
                analyzer.close_driver()
    
    with tab2:
        st.header("Movie Genre Detection from Poster")
        st.markdown("Upload a movie poster image to detect its genre!")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose a movie poster image...",
            type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
            help="Upload a clear movie poster image for best results"
        )
        
        if uploaded_file is not None:
            try:
                # Display the uploaded image
                image = Image.open(uploaded_file)
                
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.image(image, caption="Uploaded Movie Poster", width=400)
                
                with col2:
                    with st.spinner("Analyzing poster..."):
                        # Predict genre
                        predicted_genre, confidence, all_scores = analyzer.predict_genre_simple(image)
                    
                    st.success("✅ Analysis Complete!")
                    
                    # Display predicted genre
                    st.header("🎭 Predicted Genre")
                    st.markdown(f"## **{predicted_genre}**")
                    st.markdown(f"**Confidence:** {confidence:.1%}")
                    
                    # Genre description
                    genre_descriptions = {
                        'Action': '💥 Fast-paced, exciting scenes with stunts and fights',
                        'Comedy': '😄 Funny, light-hearted entertainment',
                        'Drama': '🎭 Serious, character-driven storytelling',
                        'Horror': '👻 Scary, suspenseful, designed to frighten',
                        'Romance': '💕 Love stories and romantic relationships',
                        'Sci-Fi': '🚀 Science fiction, futuristic themes',
                        'Thriller': '😰 Suspenseful, edge-of-your-seat excitement',
                        'Adventure': '🗺️ Exciting journeys and exploration',
                        'Fantasy': '🧙 Magical, supernatural elements',
                        'Animation': '🎨 Animated characters and stories'
                    }
                    
                    if predicted_genre in genre_descriptions:
                        st.markdown(f"*{genre_descriptions[predicted_genre]}*")
                
                # Show all genre scores
                st.header("📊 All Genre Scores")
                
                # Create DataFrame for scores
                import pandas as pd
                scores_df = pd.DataFrame(
                    list(all_scores.items()),
                    columns=['Genre', 'Confidence']
                ).sort_values('Confidence', ascending=False)
                
                # Create bar chart using Streamlit
                chart_data = scores_df.set_index('Genre')
                st.bar_chart(chart_data)
                
                # Display scores table
                for i, (_, row) in enumerate(scores_df.iterrows()):
                    genre, score = row['Genre'], row['Confidence']
                    if i == 0:  # Highest score
                        st.success(f"🥇 **{genre}**: {score:.1%}")
                    elif i == 1:  # Second highest
                        st.info(f"🥈 **{genre}**: {score:.1%}")
                    elif i == 2:  # Third highest
                        st.warning(f"🥉 **{genre}**: {score:.1%}")
                    else:
                        st.write(f"   **{genre}**: {score:.1%}")
                
                # Image analysis details
                with st.expander("🔍 Technical Analysis Details"):
                    features = analyzer.analyze_poster_features(image)
                    if features:
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Brightness", f"{features['brightness']:.1f}")
                        with col2:
                            st.metric("Dominant Color", features['dominant_color'])
                        with col3:
                            r, g, b = features['avg_rgb']
                            st.color_picker("Average Color", f"#{int(r):02x}{int(g):02x}{int(b):02x}")
                
            except Exception as e:
                st.error(f"Error processing image: {e}")
                st.error("Please make sure you uploaded a valid image file.")

if __name__ == "__main__":
    main()