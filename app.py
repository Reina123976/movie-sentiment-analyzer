# Page configuration - MUST be first Streamlit command
import streamlit as st
st.set_page_config(
    page_title="Movie Analyzer - Sentiment & Genre Detection",
    page_icon="🎬",
    layout="wide"
)

import plotly.express as px
import pandas as pd
import time
import pickle
import numpy as np
import re
from PIL import Image

# Handle OpenCV import with error handling
try:
    import cv2
    CV2_AVAILABLE = True
    CV2_AVAILABLE = False
    st.warning("OpenCV not available. Genre detection will use basic analysis.")

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
    st.warning("Selenium not available. Sentiment analysis will be limited.")

# Handle sklearn imports
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

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
            # Convert to numpy array
            img_array = np.array(image)
            
            # Calculate brightness
            if len(img_array.shape) == 3:
                # RGB image
                brightness = np.mean(img_array)
                
                # Calculate average colors
                avg_color = np.mean(img_array.reshape(-1, 3), axis=0)
                r, g, b = avg_color
            else:
                # Grayscale image
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
            # Convert PIL to OpenCV format
            img_array = np.array(image)
            if len(img_array.shape) == 3:
                img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            else:
                img_cv = img_array
            
            # Calculate brightness
            gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
            brightness = np.mean(gray)
            
            # Calculate dominant colors
            img_small = cv2.resize(img_cv, (50, 50))
            img_small = img_small.reshape((-1, 3))
            
            # Get average colors in BGR
            avg_color = np.mean(img_small, axis=0)
            b, g, r = avg_color
            
            # Determine dominant color category
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
            st.error(f"Error analyzing image with OpenCV: {e}")
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
        """Try to load different types of models or create a simple one"""
        try:
            # Try to load scikit-learn model
            with open('sentiment_model.pkl', 'rb') as f:
                self.sentiment_model = pickle.load(f)
            with open('vectorizer.pkl', 'rb') as f:
                self.vectorizer = pickle.load(f)
            return "sklearn"
        except:
            pass
        
        try:
            # Try to load any pickle file that might be the model
            import joblib
            self.sentiment_model = joblib.load('model.pkl')
            self.vectorizer = joblib.load('tokenizer.pkl')
            return "joblib"
        except:
            pass
        
        # Create simple rule-based model
        return "simple"
    
    def setup_driver(self, headless=True):
        """Setup Selenium Chrome driver"""
        if not SELENIUM_AVAILABLE:
            st.error("Selenium not available. Please install selenium and webdriver-manager.")
            return False
            
        try:
            chrome_options = Options()
            
            if headless:
                chrome_options.add_argument("--headless")
            
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
            chrome_options.add_argument("--window-size=1920,1080")
            chrome_options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebDriver/537.36")
            chrome_options.add_argument("--log-level=3")
            chrome_options.add_experimental_option('excludeSwitches', ['enable-logging'])
            
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
            st.error("Web scraping not available. Please install selenium.")
            return []
            
        if not movie_id.startswith('tt') or len(movie_id) < 9:
            st.error("Invalid movie ID format. Please use format like 'tt0111161'")
            return []
        
        url = f"https://www.imdb.com/title/{movie_id}/reviews"
        
        try:
            self.driver.get(url)
            time.sleep(3)
            
            if "404" in self.driver.title or "not found" in self.driver.title.lower():
                st.error("Movie not found")
                return []
            
            review_titles = []
            
            # Wait for reviews to load
            try:
                WebDriverWait(self.driver, 10).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, ".review-container, [data-testid='review-card']"))
                )
            except TimeoutException:
                pass
            
            # Try different selectors to find review titles
            selectors_to_try = [
                ".review-container .title",
                "[data-testid='review-summary-title']", 
                ".titleReviewBarItem .title",
                "h3 a",
                ".review-item .title"
            ]
            
            for selector in selectors_to_try:
                elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
                
                for element in elements[:max_reviews]:
                    try:
                        text = element.text.strip()
                        if text and len(text) > 5 and text not in review_titles:
                            review_titles.append(text)
                    except:
                        continue
                
                if review_titles:
                    break
            
            # Fallback: broader search
            if not review_titles:
                all_links = self.driver.find_elements(By.TAG_NAME, "a")
                
                for link in all_links:
                    try:
                        text = link.text.strip()
                        if (text and 15 <= len(text) <= 150 and 
                            not any(word in text.lower() for word in ['imdb', 'menu', 'sign', 'help', 'search'])):
                            review_titles.append(text)
                            if len(review_titles) >= max_reviews:
                                break
                    except:
                        continue
            
            return review_titles
            
        except Exception as e:
            st.error(f"Error scraping reviews: {e}")
            return []
    
    def predict_sentiment(self, review_text, model_type):
        """Predict sentiment based on available model type"""
        try:
            if model_type == "sklearn" and self.sentiment_model and self.vectorizer and SKLEARN_AVAILABLE:
                # Scikit-learn model
                text_vectorized = self.vectorizer.transform([review_text])
                prediction = self.sentiment_model.predict_proba(text_vectorized)[0]
                sentiment = "positive" if prediction[1] > 0.5 else "negative"
                confidence = max(prediction)
                return sentiment, confidence
                
            elif model_type == "joblib" and self.sentiment_model:
                # Try to use joblib model (adapt as needed)
                return "positive", 0.7  # Placeholder
                
            else:
                # Simple rule-based
                positive_words, negative_words = self.create_simple_sentiment_model()
                return self.predict_sentiment_simple(review_text, positive_words, negative_words)
                
        except Exception as e:
            # Fallback to simple method
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
        """Close the web driver"""
        if self.driver:
            self.driver.quit()

def main():
    st.title("🎬 Movie Analyzer")
    st.markdown("**Analyze movie reviews sentiment OR detect genre from movie poster!**")
    
    # Show available features
    col1, col2 = st.columns(2)
    with col1:
        if SELENIUM_AVAILABLE:
            st.success("✅ Web scraping available")
        else:
            st.warning("⚠️ Web scraping unavailable (install selenium)")
    
    with col2:
        if CV2_AVAILABLE:
            st.success("✅ Advanced image analysis available")
        else:
            st.warning("⚠️ Basic image analysis only (install opencv-python)")
    
    # Initialize the analyzer
    analyzer = MovieAnalyzer()
    
    # Create tabs for different functionalities
    tab1, tab2 = st.tabs(["📊 Sentiment Analysis", "🎭 Genre Detection"])
    
    with tab1:
        st.header("Movie Review Sentiment Analysis")
        
        if not SELENIUM_AVAILABLE:
            st.error("**Sentiment analysis requires web scraping capabilities.**")
            st.markdown("Please install the required packages:")
            st.code("pip install selenium webdriver-manager", language="bash")
            return
        
        st.markdown("Enter an IMDb movie ID to analyze reviews and get a recommendation!")
        
        # Load sentiment model
        with st.spinner("Loading sentiment analysis model..."):
            model_type = analyzer.load_model_alternative()
            
        if model_type == "simple":
            st.info("🤖 Using rule-based sentiment analysis. For better accuracy, train a machine learning model!")
        else:
            st.success(f"✅ Loaded {model_type} model successfully!")
        
        # Input section for sentiment analysis
        col1, col2 = st.columns([2, 1])
        
        with col1:
            movie_id = st.text_input(
                "Enter IMDb Movie ID:", 
                placeholder="e.g., tt0111161 (The Shawshank Redemption)",
                help="Find the movie ID from the IMDb URL (e.g., https://www.imdb.com/title/tt0111161/)"
            )
        
        with col2:
            max_reviews = st.slider("Max reviews to analyze", 10, 50, 30)
        
        if st.button("🔍 Analyze Movie Reviews", type="primary"):
            if not movie_id:
                st.warning("Please enter a movie ID")
                return
            
            # Setup driver
            with st.spinner("Setting up web scraper..."):
                if not analyzer.setup_driver():
                    st.error("Failed to setup web scraper")
                    return
            
            try:
                # Scrape reviews
                with st.spinner(f"Scraping reviews for {movie_id}..."):
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
                
                # Pie chart
                fig = px.pie(
                    values=[positive_count, negative_count],
                    names=['Positive', 'Negative'],
                    title="Sentiment Distribution",
                    color_discrete_map={'Positive': '#2E8B57', 'Negative': '#DC143C'}
                )
                fig.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig, use_container_width=True)
                
                # Recommendation
                st.header("🎯 Recommendation")
                
                if positive_percentage >= 70:
                    st.success("🍿 **HIGHLY RECOMMENDED!** This movie has overwhelmingly positive reviews. Go watch it!")
                elif positive_percentage >= 50:
                    st.info("👍 **RECOMMENDED** This movie has mostly positive reviews. Worth watching!")
                elif positive_percentage >= 30:
                    st.warning("🤔 **MIXED REVIEWS** This movie has mixed reception. Watch if you're curious!")
                else:
                    st.error("👎 **NOT RECOMMENDED** This movie has mostly negative reviews. You might want to skip this one.")
                
                # Detailed results
                with st.expander("📝 Detailed Review Analysis"):
                    df = pd.DataFrame(results)
                    df['confidence'] = df['confidence'].round(3)
                    df.index = df.index + 1
                    st.dataframe(
                        df,
                        use_container_width=True,
                        column_config={
                            "review": st.column_config.TextColumn("Review", width="large"),
                            "sentiment": st.column_config.TextColumn("Sentiment", width="small"),
                            "confidence": st.column_config.NumberColumn("Confidence", width="small")
                        }
                    )
                
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
                    st.image(image, caption="Uploaded Movie Poster", use_container_width=True)
                
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
                scores_df = pd.DataFrame(
                    list(all_scores.items()),
                    columns=['Genre', 'Confidence']
                ).sort_values('Confidence', ascending=False)
                
                # Create bar chart
                fig_bar = px.bar(
                    scores_df,
                    x='Genre',
                    y='Confidence',
                    title='Genre Prediction Scores',
                    color='Confidence',
                    color_continuous_scale='viridis'
                )
                fig_bar.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig_bar, use_container_width=True)
                
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
                st.error("Please make sure you uploaded a valid image file.")

if __name__ == "__main__":
    main()