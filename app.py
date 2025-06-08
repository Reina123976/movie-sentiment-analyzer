# Page configuration - MUST be first Streamlit command
import streamlit as st
st.set_page_config(
    page_title="Movie Analyzer - Sentiment & Genre Detection",
    page_icon="🎬",
    layout="wide"
)

import pandas as pd
import time
import numpy as np
import re
from PIL import Image

# Handle Plotly import with fallback
try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    st.warning("Plotly not available. Using matplotlib for charts.")
    try:
        import matplotlib.pyplot as plt
        MATPLOTLIB_AVAILABLE = True
    except ImportError:
        MATPLOTLIB_AVAILABLE = False

# Handle OpenCV import with error handling
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

# Handle Selenium imports
try:
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.chrome.service import Service
    from selenium.webdriver.chrome.options import Options
    from selenium.common.exceptions import TimeoutException
    from webdriver_manager.chrome import ChromeDriverManager
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False

class MovieAnalyzer:
    def __init__(self):
        self.driver = None
        self.sentiment_model = None
        self.vectorizer = None
        
    def create_simple_sentiment_model(self):
        """Create a simple rule-based sentiment analyzer"""
        positive_words = [
            'amazing', 'excellent', 'fantastic', 'great', 'wonderful', 'brilliant', 
            'outstanding', 'superb', 'magnificent', 'incredible', 'awesome', 'perfect',
            'beautiful', 'stunning', 'captivating', 'engaging', 'entertaining', 'thrilling',
            'exciting', 'impressive', 'remarkable', 'exceptional', 'marvelous', 'spectacular',
            'good', 'nice', 'lovely', 'enjoyable', 'fun', 'interesting', 'compelling'
        ]
        
        negative_words = [
            'terrible', 'awful', 'horrible', 'bad', 'worst', 'disappointing', 'boring',
            'dull', 'poor', 'weak', 'mediocre', 'uninspiring', 'uninteresting', 'confusing',
            'pointless', 'ridiculous', 'stupid', 'waste', 'annoying', 'frustrating',
            'disappointing', 'predictable', 'cliche', 'overrated', 'fail', 'lacking'
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
    
    def analyze_poster_features_basic(self, image):
        """Basic image analysis without OpenCV"""
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
            if r > g and r > b:
                if r > 150:
                    dominant_color = 'red'
                else:
                    dominant_color = 'dark'
            elif g > r and g > b:
                dominant_color = 'green'
            elif b > r and b > g:
                if b > 150:
                    dominant_color = 'blue'
                else:
                    dominant_color = 'dark'
            else:
                if brightness > 150:
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
    
    def analyze_poster_features_opencv(self, image):
        """Advanced image analysis with OpenCV"""
        try:
            img_array = np.array(image)
            if len(img_array.shape) == 3:
                img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            else:
                img_cv = img_array
            
            gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
            brightness = np.mean(gray)
            
            img_small = cv2.resize(img_cv, (50, 50))
            img_small = img_small.reshape((-1, 3))
            
            avg_color = np.mean(img_small, axis=0)
            b, g, r = avg_color
            
            if r > g and r > b:
                if r > 150:
                    dominant_color = 'red'
                else:
                    dominant_color = 'dark'
            elif g > r and g > b:
                dominant_color = 'green'
            elif b > r and b > g:
                if b > 150:
                    dominant_color = 'blue'
                else:
                    dominant_color = 'dark'
            else:
                if brightness > 150:
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
            return self.analyze_poster_features_basic(image)
    
    def analyze_poster_features(self, image):
        """Analyze poster features using available libraries"""
        if CV2_AVAILABLE:
            return self.analyze_poster_features_opencv(image)
        else:
            return self.analyze_poster_features_basic(image)
    
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
        
        for genre, characteristics in genre_colors.items():
            score = 0.5
            
            if characteristics['brightness'] == 'high' and brightness > 150:
                score += 0.3
            elif characteristics['brightness'] == 'medium' and 80 <= brightness <= 150:
                score += 0.2
            elif characteristics['brightness'] == 'low' and brightness < 100:
                score += 0.3
            elif characteristics['brightness'] == 'very_low' and brightness < 60:
                score += 0.4
            
            if dominant_color in [color.lower() for color in characteristics['dominant_colors']]:
                score += 0.3
            
            if genre == 'Horror' and brightness < 70 and dominant_color in ['dark', 'red']:
                score += 0.2
            elif genre == 'Comedy' and brightness > 140:
                score += 0.2
            elif genre == 'Sci-Fi' and dominant_color == 'blue':
                score += 0.2
            
            genre_scores[genre] = min(0.95, score)
        
        predicted_genre = max(genre_scores, key=genre_scores.get)
        confidence = genre_scores[predicted_genre]
        
        return predicted_genre, confidence, genre_scores
    
    def create_plotly_pie_chart(self, positive_count, negative_count):
        """Create pie chart with Plotly"""
        if PLOTLY_AVAILABLE:
            fig = px.pie(
                values=[positive_count, negative_count],
                names=['Positive', 'Negative'],
                title="Sentiment Distribution",
                color_discrete_map={'Positive': '#2E8B57', 'Negative': '#DC143C'}
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            return fig
        else:
            return None
    
    def create_plotly_bar_chart(self, scores_df):
        """Create bar chart with Plotly"""
        if PLOTLY_AVAILABLE:
            fig = px.bar(
                scores_df,
                x='Genre',
                y='Confidence',
                title='Genre Prediction Scores',
                color='Confidence',
                color_continuous_scale='viridis'
            )
            fig.update_layout(xaxis_tickangle=-45)
            return fig
        else:
            return None
    
    def create_matplotlib_pie_chart(self, positive_count, negative_count):
        """Create pie chart with matplotlib as fallback"""
        if MATPLOTLIB_AVAILABLE:
            fig, ax = plt.subplots()
            sizes = [positive_count, negative_count]
            labels = ['Positive', 'Negative']
            colors = ['#2E8B57', '#DC143C']
            ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
            ax.set_title('Sentiment Distribution')
            return fig
        else:
            return None
    
    def create_matplotlib_bar_chart(self, scores_df):
        """Create bar chart with matplotlib as fallback"""
        if MATPLOTLIB_AVAILABLE:
            fig, ax = plt.subplots(figsize=(10, 6))
            bars = ax.bar(scores_df['Genre'], scores_df['Confidence'])
            ax.set_title('Genre Prediction Scores')
            ax.set_xlabel('Genre')
            ax.set_ylabel('Confidence')
            plt.xticks(rotation=45)
            plt.tight_layout()
            return fig
        else:
            return None
    
    def setup_driver(self, headless=True):
        """Setup Selenium Chrome driver"""
        if not SELENIUM_AVAILABLE:
            return False
            
        try:
            chrome_options = Options()
            
            if headless:
                chrome_options.add_argument("--headless")
            
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
            chrome_options.add_argument("--window-size=1920,1080")
            chrome_options.add_argument("--disable-gpu")
            chrome_options.add_argument("--disable-dev-tools")
            chrome_options.add_argument("--no-zygote")
            chrome_options.add_argument("--single-process")
            
            service = Service(ChromeDriverManager().install())
            self.driver = webdriver.Chrome(service=service, options=chrome_options)
            self.driver.implicitly_wait(10)
            self.driver.set_page_load_timeout(30)
            
            return True
        except Exception as e:
            st.error(f"Error setting up web driver: {e}")
            return False
    
    def scrape_reviews(self, movie_id, max_reviews=30):
        """Scrape review titles from IMDb"""
        if not SELENIUM_AVAILABLE:
            return []
            
        if not movie_id.startswith('tt') or len(movie_id) < 9:
            st.error("Invalid movie ID format")
            return []
        
        # For cloud deployment, web scraping might be limited
        # Return sample data for demonstration
        sample_reviews = [
            "Amazing storyline and great acting",
            "Brilliant cinematography but slow pacing",
            "Wonderful performances by the cast",
            "Disappointing sequel to a great movie",
            "Excellent direction and beautiful visuals",
            "Boring and predictable plot",
            "Outstanding special effects",
            "Poor character development",
            "Engaging and entertaining throughout",
            "Waste of time and money"
        ]
        
        st.info("🔄 Web scraping may be limited in cloud environment. Using sample reviews for demonstration.")
        return sample_reviews[:max_reviews//3]  # Return fewer sample reviews
    
    def predict_sentiment(self, review_text):
        """Predict sentiment using rule-based approach"""
        positive_words, negative_words = self.create_simple_sentiment_model()
        return self.predict_sentiment_simple(review_text, positive_words, negative_words)
    
    def analyze_reviews(self, reviews):
        """Analyze sentiment for all reviews"""
        results = []
        progress_bar = st.progress(0)
        
        for i, review in enumerate(reviews):
            sentiment, confidence = self.predict_sentiment(review)
            results.append({
                'review': review,
                'sentiment': sentiment,
                'confidence': confidence
            })
            progress_bar.progress((i + 1) / len(reviews))
        
        return results
    
    def close_driver(self):
        """Close the web driver"""
        if self.driver:
            self.driver.quit()

def main():
    st.title("🎬 Movie Analyzer")
    st.markdown("**Analyze movie reviews sentiment OR detect genre from movie poster!**")
    
    # Show available features
    col1, col2, col3 = st.columns(3)
    with col1:
        if PLOTLY_AVAILABLE:
            st.success("✅ Plotly charts available")
        elif MATPLOTLIB_AVAILABLE:
            st.info("📊 Matplotlib charts available")
        else:
            st.warning("⚠️ Limited chart functionality")
    
    with col2:
        if CV2_AVAILABLE:
            st.success("✅ Advanced image analysis")
        else:
            st.info("📷 Basic image analysis")
    
    with col3:
        if SELENIUM_AVAILABLE:
            st.warning("⚠️ Web scraping limited in cloud")
        else:
            st.warning("⚠️ No web scraping available")
    
    # Initialize the analyzer
    analyzer = MovieAnalyzer()
    
    # Create tabs for different functionalities
    tab1, tab2 = st.tabs(["📊 Sentiment Analysis", "🎭 Genre Detection"])
    
    with tab1:
        st.header("Movie Review Sentiment Analysis")
        st.markdown("Enter an IMDb movie ID to analyze reviews and get a recommendation!")
        
        # Input section for sentiment analysis
        col1, col2 = st.columns([2, 1])
        
        with col1:
            movie_id = st.text_input(
                "Enter IMDb Movie ID:", 
                placeholder="e.g., tt0111161 (The Shawshank Redemption)",
                help="In cloud environment, this uses sample reviews for demonstration"
            )
        
        with col2:
            max_reviews = st.slider("Max reviews to analyze", 5, 20, 10)
        
        if st.button("🔍 Analyze Movie Reviews", type="primary"):
            if not movie_id:
                st.warning("Please enter a movie ID")
                return
            
            try:
                # Get reviews (sample data in cloud)
                with st.spinner("Getting reviews..."):
                    reviews = analyzer.scrape_reviews(movie_id, max_reviews)
                
                if not reviews:
                    st.warning("No reviews found")
                    return
                
                st.success(f"Found {len(reviews)} reviews!")
                
                # Analyze sentiment
                with st.spinner("Analyzing sentiment..."):
                    results = analyzer.analyze_reviews(reviews)
                
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
                
                # Charts
                plotly_fig = analyzer.create_plotly_pie_chart(positive_count, negative_count)
                if plotly_fig:
                    st.plotly_chart(plotly_fig, use_container_width=True)
                else:
                    matplotlib_fig = analyzer.create_matplotlib_pie_chart(positive_count, negative_count)
                    if matplotlib_fig:
                        st.pyplot(matplotlib_fig)
                    else:
                        # Fallback to simple text display
                        st.write(f"**Results:** {positive_count} positive, {negative_count} negative")
                
                # Recommendation
                st.header("🎯 Recommendation")
                
                if positive_percentage >= 70:
                    st.success("🍿 **HIGHLY RECOMMENDED!** This movie has overwhelmingly positive reviews!")
                elif positive_percentage >= 50:
                    st.info("👍 **RECOMMENDED** This movie has mostly positive reviews!")
                elif positive_percentage >= 30:
                    st.warning("🤔 **MIXED REVIEWS** This movie has mixed reception!")
                else:
                    st.error("👎 **NOT RECOMMENDED** This movie has mostly negative reviews!")
                
                # Detailed results
                with st.expander("📝 Detailed Review Analysis"):
                    df = pd.DataFrame(results)
                    df['confidence'] = df['confidence'].round(3)
                    df.index = df.index + 1
                    st.dataframe(df, use_container_width=True)
                
            except Exception as e:
                st.error(f"An error occurred: {e}")
            
            finally:
                analyzer.close_driver()
    
    with tab2:
        st.header("Movie Genre Detection from Poster")
        st.markdown("Upload a movie poster image to detect its genre!")
        
        uploaded_file = st.file_uploader(
            "Choose a movie poster image...",
            type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
            help="Upload a clear movie poster image for best results"
        )
        
        if uploaded_file is not None:
            try:
                image = Image.open(uploaded_file)
                
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.image(image, caption="Uploaded Movie Poster", use_container_width=True)
                
                with col2:
                    with st.spinner("Analyzing poster..."):
                        predicted_genre, confidence, all_scores = analyzer.predict_genre_simple(image)
                    
                    st.success("✅ Analysis Complete!")
                    
                    st.header("🎭 Predicted Genre")
                    st.markdown(f"## **{predicted_genre}**")
                    st.markdown(f"**Confidence:** {confidence:.1%}")
                    
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
                
                scores_df = pd.DataFrame(
                    list(all_scores.items()),
                    columns=['Genre', 'Confidence']
                ).sort_values('Confidence', ascending=False)
                
                # Try Plotly first, then matplotlib, then simple display
                plotly_bar = analyzer.create_plotly_bar_chart(scores_df)
                if plotly_bar:
                    st.plotly_chart(plotly_bar, use_container_width=True)
                else:
                    matplotlib_bar = analyzer.create_matplotlib_bar_chart(scores_df)
                    if matplotlib_bar:
                        st.pyplot(matplotlib_bar)
                    else:
                        st.bar_chart(scores_df.set_index('Genre'))
                
                # Display scores table
                st.dataframe(
                    scores_df,
                    use_container_width=True,
                    column_config={
                        "Genre": st.column_config.TextColumn("Genre"),
                        "Confidence": st.column_config.ProgressColumn("Confidence", min_value=0, max_value=1)
                    }
                )
                
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

if __name__ == "__main__":
    main()