from flask import Flask, render_template, send_from_directory, request, jsonify
import pandas as pd
import os
import random
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from PIL import Image
import numpy as np

def standardize_category(cat):
    """Maps specific clothing labels to broad categories like Shirt, T-shirt, Pants, etc."""
    cat = str(cat).lower().strip()
    if 't-shirt' in cat or 'tshirt' in cat or 'top' in cat or 'undershirt' in cat:
        return 'T-shirt'
    if 'shirt' in cat or 'blouse' in cat or 'longsleeve' in cat or 'polo' in cat:
        return 'Shirt'
    if 'pant' in cat or 'jeans' in cat or 'trouser' in cat or 'short' in cat:
        return 'Pants'
    if 'jacket' in cat or 'coat' in cat or 'blazer' in cat or 'hoodie' in cat or 'outwear' in cat or 'sweater' in cat:
        return 'Jacket'
    if 'dress' in cat or 'skirt' in cat:
        return 'Dress'
    if 'shoe' in cat:
        return 'Shoes'
    if 'hat' in cat or 'bag' in cat or 'accessory' in cat:
        return 'Accessories'
    return 'Other'

app = Flask(__name__)

# Train ML model on startup using only Clothes Price Prediction Dataset
try:
    ds1_path = os.path.join(os.path.dirname(__file__), 'Clothes Price Prediction Dataset.csv')
    df1 = pd.read_csv(ds1_path)
    
    # Clean the dataset
    combined_df = pd.DataFrame({
        'Brand': df1['Brand'].astype(str),
        'Category': df1['Category'].apply(standardize_category),
        'Color': df1['Color'].astype(str),
        'Size': df1['Size'].astype(str),
        'Material': df1['Material'].astype(str),
        'Price': df1['Price']
    })
    
    # Extract unique brands, ignore 'nan'
    all_brands = sorted([b for b in combined_df['Brand'].unique() if b.lower() != 'nan'])
    
    X = combined_df[['Brand', 'Category', 'Color', 'Size', 'Material']]
    y = combined_df['Price']

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), ['Brand', 'Category', 'Color', 'Size', 'Material'])
        ])

    model = Pipeline(steps=[('preprocessor', preprocessor),
                            ('regressor', RandomForestRegressor(n_estimators=30, random_state=42))])
    
    model.fit(X, y)
    print("Model trained successfully!")
    
    # Global categories and colors from dataset 1 (focused list)
    all_brands = sorted([b for b in combined_df['Brand'].unique() if str(b).lower() != 'nan'])
    
    # Ensure all broad categories are available even if the small price dataset is missing some
    standard_categories = {'Shirt', 'T-shirt', 'Pants', 'Jacket', 'Dress', 'Shoes', 'Accessories'}
    all_categories = sorted(list(set(combined_df['Category'].unique()) | standard_categories))
    all_categories = [c for c in all_categories if str(c).lower() != 'nan' and c != 'Other']
    
    all_colors = sorted([c for c in combined_df['Color'].unique() if str(c).lower() != 'nan'])
except Exception as e:
    model = None
    all_brands = []
    print(f"Error training model: {e}")

# Initialize models as None
brand_clf, cat_clf, color_clf, quality_clf = None, None, None, None

# Train Image Analysis models
image_models_ready = False
try:
    # 1. Brand Detection (Logos)
    logo_dir = os.path.join(os.path.dirname(__file__), 'Logos')
    X_logos, y_logos = [], []
    if os.path.exists(logo_dir):
        for filename in os.listdir(logo_dir):
            if filename.endswith(('.jpg', '.png', '.jpeg')):
                brand_name = os.path.splitext(filename)[0]
                img_path = os.path.join(logo_dir, filename)
                try:
                    with Image.open(img_path) as img:
                        img = img.convert('RGB').resize((32, 32))
                        # Original
                        X_logos.append(np.array(img).flatten())
                        y_logos.append(brand_name)
                        
                        # Inverted (helps detect white logos on dark clothes)
                        from PIL import ImageOps
                        inverted_img = ImageOps.invert(img)
                        X_logos.append(np.array(inverted_img).flatten())
                        y_logos.append(brand_name)
                except: continue
        
    if X_logos:
        # Harvest real product samples for Nike and Puma from the fashion dataset to improve context
        styles_path = os.path.join(os.path.dirname(__file__), 'fashion Product Images Dataset', 'styles.csv')
        image_dir = os.path.join(os.path.dirname(__file__), 'fashion Product Images Dataset', 'images')
        if os.path.exists(styles_path) and os.path.exists(image_dir):
            try:
                styles_df = pd.read_csv(styles_path, on_bad_lines='skip')
                for brand in ['Nike', 'Puma']:
                    brand_samples = styles_df[styles_df['productDisplayName'].str.contains(brand, case=False, na=False)].head(50)
                    for _, row in brand_samples.iterrows():
                        img_path = os.path.join(image_dir, str(row['id']) + '.jpg')
                        if os.path.exists(img_path):
                            try:
                                with Image.open(img_path) as img:
                                    img = img.convert('RGB').resize((32, 32))
                                    X_logos.append(np.array(img).flatten())
                                    y_logos.append(brand)
                            except: continue
            except: pass

        from sklearn.neighbors import KNeighborsClassifier
        brand_clf = KNeighborsClassifier(n_neighbors=3) # Increase neighbors for better voting
        brand_clf.fit(X_logos, y_logos)
        print("Brand model (Logos + Samples) trained!")

    # 2. Category Detection (Catergory Analyzer Datset + Fashion Dataset Supplement)
    cat_ds_dir = os.path.join(os.path.dirname(__file__), 'Catergory Analyzer Datset')
    cat_csv_path = os.path.join(cat_ds_dir, 'images.csv')
    cat_img_dir = os.path.join(cat_ds_dir, 'images')
    
    X_cat, y_cat_labels = [], []

    if os.path.exists(cat_csv_path) and os.path.exists(cat_img_dir):
        cat_df = pd.read_csv(cat_csv_path)
        cat_df = cat_df.dropna(subset=['image', 'label'])
        
        # Increased sample size for better accuracy
        cat_sample = cat_df.sample(n=min(1000, len(cat_df)), random_state=42)
        
        for _, row in cat_sample.iterrows():
            img_path = os.path.join(cat_img_dir, row['image'] + '.jpg')
            if not os.path.exists(img_path):
                img_path = os.path.join(cat_img_dir, row['image'] + '.png')
                
            if os.path.exists(img_path):
                try:
                    with Image.open(img_path) as img:
                        img = img.convert('RGB').resize((32, 32))
                        X_cat.append(np.array(img).flatten())
                        y_cat_labels.append(standardize_category(row['label']))
                except: continue

    # Supplement with high-quality samples from the main fashion dataset
    styles_path = os.path.join(os.path.dirname(__file__), 'fashion Product Images Dataset', 'styles.csv')
    fashion_img_dir = os.path.join(os.path.dirname(__file__), 'fashion Product Images Dataset', 'images')
    if os.path.exists(styles_path) and os.path.exists(fashion_img_dir):
        try:
            f_df = pd.read_csv(styles_path, on_bad_lines='skip')
            # Select 100 clear samples for each major category
            for cat_name in ['Tshirts', 'Shirts', 'Trousers', 'Jeans', 'Dresses', 'Shoes']:
                samples = f_df[f_df['articleType'] == cat_name].head(100)
                for _, row in samples.iterrows():
                    img_path = os.path.join(fashion_img_dir, str(row['id']) + '.jpg')
                    if os.path.exists(img_path):
                        try:
                            with Image.open(img_path) as img:
                                img = img.convert('RGB').resize((32, 32))
                                X_cat.append(np.array(img).flatten())
                                y_cat_labels.append(standardize_category(cat_name))
                        except: continue
        except: pass
        
    if X_cat:
        from sklearn.neighbors import KNeighborsClassifier
        cat_clf = KNeighborsClassifier(n_neighbors=5)
        cat_clf.fit(X_cat, y_cat_labels)
        print("Category model (Hybrid + KNN) trained!")
            
    # 3. Color Detection
    styles_path = os.path.join(os.path.dirname(__file__), 'fashion Product Images Dataset', 'styles.csv')
    if os.path.exists(styles_path):
        styles_df = pd.read_csv(styles_path, on_bad_lines='skip')
        image_dir = os.path.join(os.path.dirname(__file__), 'fashion Product Images Dataset', 'images')
        if os.path.exists(image_dir):
            available_images = set(os.listdir(image_dir))
            styles_df['filename'] = styles_df['id'].astype(str) + '.jpg'
            styles_df = styles_df[styles_df['filename'].isin(available_images)]
            color_sample = styles_df.sample(n=min(200, len(styles_df)), random_state=42)
            X_color, y_color_labels = [], []
            for _, row in color_sample.iterrows():
                try:
                    with Image.open(os.path.join(image_dir, row['filename'])) as img:
                        img = img.convert('RGB').resize((32, 32))
                        X_color.append(np.array(img).flatten())
                        y_color_labels.append(row['baseColour'])
                except: continue
            
            if X_color:
                color_clf = RandomForestClassifier(n_estimators=20, random_state=42)
                color_clf.fit(X_color, y_color_labels)
                print("Color model trained!")
    
    # 4. Quality Detection (Fabric Defect Dataset)
    defect_dir = os.path.join(os.path.dirname(__file__), 'Fabric Defect')
    X_quality, y_quality = [], []
    
    if os.path.exists(defect_dir):
        # We'll map defects to quality levels
        # Holes are major defects (Poor), Lines are moderate (Fair)
        defect_mapping = {
            'hole': 'Poor',
            'horizontal': 'Fair',
            'verticle': 'Fair'
        }
        
        for d_folder, label in defect_mapping.items():
            folder_path = os.path.join(defect_dir, d_folder)
            if os.path.exists(folder_path):
                files = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.png', '.jpeg'))]
                # Take more samples for better coverage
                for filename in random.sample(files, min(100, len(files))):
                    try:
                        with Image.open(os.path.join(folder_path, filename)) as img:
                            img = img.convert('RGB').resize((32, 32))
                            X_quality.append(np.array(img).flatten())
                            y_quality.append(label)
                    except: continue
        
        # Add high-resolution "captured" defects which are more representative of real-world tears
        captured_hole_dir = os.path.join(defect_dir, 'captured', 'Hole')
        if os.path.exists(captured_hole_dir):
            files = [f for f in os.listdir(captured_hole_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
            for filename in random.sample(files, min(50, len(files))):
                try:
                    with Image.open(os.path.join(captured_hole_dir, filename)) as img:
                        img = img.convert('RGB').resize((32, 32))
                        X_quality.append(np.array(img).flatten())
                        y_quality.append('Poor') # Large holes/tears
                except: continue

        captured_lines_dir = os.path.join(defect_dir, 'captured', 'Lines')
        if os.path.exists(captured_lines_dir):
            files = [f for f in os.listdir(captured_lines_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
            for filename in random.sample(files, min(30, len(files))):
                try:
                    with Image.open(os.path.join(captured_lines_dir, filename)) as img:
                        img = img.convert('RGB').resize((32, 32))
                        X_quality.append(np.array(img).flatten())
                        y_quality.append('Fair') # Visible lines/scratches
                except: continue
        
        # Add "Excellent" and "Good" samples from the main fashion dataset - balanced with defects
        fashion_img_dir = os.path.join(os.path.dirname(__file__), 'fashion Product Images Dataset', 'images')
        if os.path.exists(fashion_img_dir):
            files = [f for f in os.listdir(fashion_img_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
            # Use a much larger sample of clean clothes to prevent bias
            clean_samples = random.sample(files, min(500, len(files)))
            for i, filename in enumerate(clean_samples):
                try:
                    with Image.open(os.path.join(fashion_img_dir, filename)) as img:
                        img = img.convert('RGB').resize((32, 32))
                        X_quality.append(np.array(img).flatten())
                        # Split between Excellent and Good
                        if i < 300:
                            y_quality.append('Excellent')
                        else:
                            y_quality.append('Good')
                except: continue

        # Explicitly add Torn.jpg if it exists (highly weighted for Poor)
        torn_path = os.path.join(defect_dir, 'Torn.jpg')
        if os.path.exists(torn_path):
            try:
                with Image.open(torn_path) as img:
                    img = img.convert('RGB').resize((32, 32))
                    for _ in range(20): # Weight it strongly
                        X_quality.append(np.array(img).flatten())
                        y_quality.append('Poor')
            except: pass
                
        if X_quality:
            # RandomForest is better at ignoring 'noise' like stripes/logos than KNN
            cat_clf_quality = RandomForestClassifier(n_estimators=50, random_state=42)
            cat_clf_quality.fit(X_quality, y_quality)
            quality_clf = cat_clf_quality
            print("Quality model (Balanced RF) trained!")
            
    image_models_ready = True

except Exception as e:
    print(f"Error training specialized image models: {e}")

# Global storage for user listings and cart
user_listings = []
cart = []
liked_ids = []
disliked_ids = []
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'static', 'uploads')
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download NLTK data
try:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('punkt_tab')
except: pass

@app.route('/')
def index():
    # Get filter parameters
    search_query = request.args.get('search', '').lower()
    f_brand = request.args.get('brand', '')
    f_category = request.args.get('category', '')
    f_color = request.args.get('color', '')
    f_min_price = request.args.get('min_price', type=float)
    f_max_price = request.args.get('max_price', type=float)

    try:
        styles_path = os.path.join(os.path.dirname(__file__), 'fashion Product Images Dataset', 'styles.csv')
        df = pd.read_csv(styles_path, on_bad_lines='skip')
        df = df.dropna(subset=['id', 'productDisplayName'])
        
        # Prepare products list
        products = []
        for p in user_listings: products.append(p)
            
        sample_df = df.head(500).copy() # Use larger sample for similarity matching
        for _, r in sample_df.iterrows():
            products.append({
                "id": str(r['id']),
                "productDisplayName": r['productDisplayName'],
                "brand": r['productDisplayName'].split()[0],
                "articleType": r['articleType'],
                "baseColour": r['baseColour'],
                "Price (INR)": round(((r['id'] % 20 + 5) * 150 - 1) * 0.4),
                "image_url": f"/images/{r['id']}.jpg",
                "is_user": False
            })

        # --- ADVANCED PERSONALIZATION ---
        liked_features = []
        if liked_ids:
            # Extract features of liked items to build a preference profile
            for lid in liked_ids:
                # Find product data for this ID
                p_data = None
                for ul in user_listings:
                    if ul['id'] == lid: p_data = ul; break
                if not p_data:
                    # Check CSV
                    try:
                        row = df[df['id'].astype(str) == lid]
                        if not row.empty:
                            r = row.iloc[0]
                            p_data = {"productDisplayName": r['productDisplayName'], "articleType": r['articleType'], "baseColour": r['baseColour']}
                    except: pass
                if p_data:
                    liked_features.append(f"{p_data['productDisplayName']} {p_data.get('articleType','')} {p_data.get('baseColour','')}".lower())

        # --- ADVANCED NLP SEARCH & RANKING ---
        # 5. Apply Preference Ranking if no search query
        if not search_query and liked_features:
            product_texts = [f"{p['productDisplayName']} {p['articleType']} {p.get('baseColour', '')} {p.get('brand','')}".lower() for p in products]
            vectorizer = TfidfVectorizer()
            tfidf_matrix = vectorizer.fit_transform(product_texts)
            
            # Create a profile vector from all liked items
            profile_vec = vectorizer.transform([" ".join(liked_features)])
            pref_sim = cosine_similarity(profile_vec, tfidf_matrix).flatten()
            
            for i, p in enumerate(products):
                p['pref_score'] = pref_sim[i]
            
            # Sort: Liked-similar items first, then original order
            # Disliked items are filtered out
            products = [p for p in products if p['id'] not in disliked_ids]
            products.sort(key=lambda x: x.get('pref_score', 0), reverse=True)

        elif search_query:
            # 1. NLP Keyword Parsing & Intent Detection
            tokens = word_tokenize(search_query.lower())
            stop_words = set(stopwords.words('english'))
            
            # Identify price intent
            price_intent = None
            if any(w in ['cheap', 'low', 'affordable', 'budget', 'under'] for w in tokens):
                price_intent = 'cheap'
            elif any(w in ['expensive', 'premium', 'luxury', 'high'] for w in tokens):
                price_intent = 'expensive'
            
            # Filter keywords for similarity (remove price words)
            price_words = {'cheap', 'low', 'affordable', 'budget', 'under', 'expensive', 'premium', 'luxury', 'high', 'price', 'cost'}
            search_keywords = [w for w in tokens if w not in stop_words and w not in price_words]
            filtered_query = " ".join(search_keywords) if search_keywords else search_query

            # 2. TF-IDF Similarity Scoring
            product_texts = [f"{p['productDisplayName']} {p['articleType']} {p.get('baseColour', '')} {p.get('brand','')}".lower() for p in products]
            vectorizer = TfidfVectorizer()
            tfidf_matrix = vectorizer.fit_transform(product_texts)
            query_vec = vectorizer.transform([filtered_query])
            
            cosine_sim = cosine_similarity(query_vec, tfidf_matrix).flatten()
            
            for i, p in enumerate(products):
                p['score'] = cosine_sim[i]
                if p['id'] in disliked_ids: p['score'] = -1

            # 3. Filter and Sort
            # Keep items with some relevance
            products = [p for p in products if p['score'] > 0.02 and p['id'] not in disliked_ids]
            
            if price_intent == 'cheap':
                # Primary: Similarity (to ensure it's a tshirt), Secondary: Price
                products.sort(key=lambda x: (-x['score'], x['Price (INR)']))
            elif price_intent == 'expensive':
                products.sort(key=lambda x: (-x['score'], -x['Price (INR)']))
            else:
                products.sort(key=lambda x: x['score'], reverse=True)

    except Exception as e:
        products = [p for p in user_listings if p['id'] not in disliked_ids][::-1]
        print(f"Search Error: {e}")
        
    return render_template('index.html', products=products, cart_count=len(cart), 
                           search=search_query, brand=f_brand, category=f_category, 
                           color=f_color, min_price=f_min_price, max_price=f_max_price,
                           all_brands=all_brands, all_categories=all_categories, all_colors=all_colors,
                           liked_ids=liked_ids, disliked_ids=disliked_ids)

@app.route('/api/like_product', methods=['POST'])
def like_product():
    pid = str(request.json.get('product_id'))
    if pid not in liked_ids:
        liked_ids.append(pid)
        if pid in disliked_ids: disliked_ids.remove(pid)
    return jsonify({"success": True, "liked": True})

@app.route('/api/dislike_product', methods=['POST'])
def dislike_product():
    pid = str(request.json.get('product_id'))
    if pid not in disliked_ids:
        disliked_ids.append(pid)
        if pid in liked_ids: liked_ids.remove(pid)
    return jsonify({"success": True, "disliked": True})

@app.route('/product/<product_id>')
def product_detail(product_id):
    product = None
    # Check user listings first
    for p in user_listings:
        if p['id'] == product_id:
            product = p
            break
    
    if not product:
        # Check CSV
        try:
            styles_path = os.path.join(os.path.dirname(__file__), 'fashion Product Images Dataset', 'styles.csv')
            df = pd.read_csv(styles_path, on_bad_lines='skip')
            row = df[df['id'].astype(str) == product_id]
            if not row.empty:
                r = row.iloc[0]
                product = {
                    "id": str(r['id']),
                    "productDisplayName": r['productDisplayName'],
                    "articleType": r['articleType'],
                    "Price (INR)": round(((r['id'] % 20 + 5) * 150 - 1) * 0.4),
                    "description": f"A beautiful {r['articleType']} piece in {r['baseColour']}.",
                    "image_url": f"/images/{r['id']}.jpg"
                }
        except: pass

    if not product:
        return "Product not found", 404
        
    # --- SIMILAR MATCHES LOGIC (Cosine Similarity) ---
    similar_products = []
    try:
        styles_path = os.path.join(os.path.dirname(__file__), 'fashion Product Images Dataset', 'styles.csv')
        df = pd.read_csv(styles_path, on_bad_lines='skip')
        df = df.dropna(subset=['id', 'productDisplayName'])
        
        # We'll use a subset of the catalog for similarity to keep it fast
        candidates = []
        # Add user listings
        for p in user_listings:
            if p['id'] != product_id: candidates.append(p)
        
        # Add a slice of CSV
        csv_sample = df.sample(100) if len(df) > 100 else df
        for _, r in csv_sample.iterrows():
            if str(r['id']) != product_id:
                candidates.append({
                    "id": str(r['id']),
                    "productDisplayName": r['productDisplayName'],
                    "articleType": r['articleType'],
                    "baseColour": r['baseColour'],
                    "Price (INR)": round(((r['id'] % 20 + 5) * 150 - 1) * 0.4),
                    "image_url": f"/images/{r['id']}.jpg"
                })

        # Feature Vectorization
        # Combined text features
        texts = [f"{p['productDisplayName']} {p['articleType']} {p.get('baseColour','')}".lower() for p in [product] + candidates]
        
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(texts)
        
        # Compute Cosine Similarity
        cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
        
        # Add scores and sort
        for i, c in enumerate(candidates):
            c['similarity_score'] = cosine_sim[i]
        
        candidates.sort(key=lambda x: x['similarity_score'], reverse=True)
        similar_products = candidates[:4] # Top 4
        
    except Exception as e:
        print(f"Similarity Error: {e}")

    is_liked = str(product_id) in liked_ids
    is_disliked = str(product_id) in disliked_ids

    return render_template('product.html', product=product, similar_products=similar_products, 
                           cart_count=len(cart), is_liked=is_liked, is_disliked=is_disliked)

@app.route('/cart')
def view_cart():
    total = sum(item['Price (INR)'] for item in cart)
    return render_template('cart.html', cart_items=cart, total=total, cart_count=len(cart))

@app.route('/api/add_to_cart', methods=['POST'])
def add_to_cart():
    product_id = request.json.get('product_id')
    # Find product
    product = None
    for p in user_listings:
        if p['id'] == product_id:
            product = p
            break
    
    if not product:
        try:
            styles_path = os.path.join(os.path.dirname(__file__), 'fashion Product Images Dataset', 'styles.csv')
            df = pd.read_csv(styles_path, on_bad_lines='skip')
            row = df[df['id'].astype(str) == product_id]
            if not row.empty:
                r = row.iloc[0]
                product = {
                    "id": str(r['id']),
                    "productDisplayName": r['productDisplayName'],
                    "Price (INR)": round(((r['id'] % 20 + 5) * 150 - 1) * 0.4),
                    "image_url": f"/images/{r['id']}.jpg"
                }
        except: pass

    if product:
        cart.append(product)
        return jsonify({"success": True, "cart_count": len(cart)})
    return jsonify({"error": "Product not found"}), 404

@app.route('/images/<filename>')
def serve_image(filename):
    image_dir = os.path.join(os.path.dirname(__file__), 'fashion Product Images Dataset', 'images')
    return send_from_directory(image_dir, filename)

@app.route('/sell')
def sell():
    return render_template('sell.html', brands=all_brands, categories=all_categories, cart_count=len(cart))

@app.route('/api/list_product', methods=['POST'])
def list_product():
    # We use request.form and request.files for multipart/form-data
    title = request.form.get('title')
    brand = request.form.get('brand')
    category = request.form.get('category')
    price = request.form.get('price')
    description = request.form.get('description', 'No description provided.')
    image = request.files.get('image')
    
    if not image:
        return jsonify({"error": "No image uploaded"}), 400
        
    try:
        # Save image to static/uploads
        ext = os.path.splitext(image.filename)[1]
        filename = f"listing_{len(user_listings)}{ext}"
        save_path = os.path.join(UPLOAD_FOLDER, filename)
        image.save(save_path)
        
        # Add to global list
        new_product = {
            "id": f"u{len(user_listings)}", # prefixed with u to distinguish
            "productDisplayName": title if title else f"{brand} {category}",
            "articleType": category,
            "Price (INR)": round(float(price)) if price else 0, # Round price
            "description": description,
            "image_url": f"/static/uploads/{filename}",
            "is_user": True
        }
        user_listings.append(new_product)
        return jsonify({"success": True})
    except Exception as e:
        print(f"Listing error: {e}")
        return jsonify({"error": str(e)}), 400

@app.route('/api/predict_price', methods=['POST'])
def predict_price():
    if not model:
        return jsonify({"error": "Model not available"}), 500
    data = request.json
    # Standardize the input category before prediction to match training data
    if 'Category' in data:
        data['Category'] = standardize_category(data['Category'])
        
    input_df = pd.DataFrame([data])
    try:
        # Remove metadata not used in the core ML model
        prediction_input = input_df.drop(columns=['Quality', 'ItemAge'], errors='ignore')
        prediction = model.predict(prediction_input)[0]
        
        # 1. Quality Multiplier
        quality_multipliers = {"Excellent": 0.5, "Good": 0.4, "Fair": 0.25, "Poor": 0.1}
        quality = data.get('Quality', 'Excellent')
        q_multiplier = quality_multipliers.get(quality, 0.5)
        
        # 2. Age Depreciation (10% reduction per year, max 80% total reduction)
        age = float(data.get('ItemAge', 0))
        age_multiplier = max(0.2, 1.0 - (age * 0.1))
        
        final_price = float(prediction * q_multiplier * age_multiplier)
        return jsonify({"price": final_price})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/api/analyze_image', methods=['POST'])
def analyze_image():
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400
    file = request.files['image']
    try:
        img = Image.open(file.stream).convert('RGB').resize((32, 32))
        features = np.array(img).flatten().reshape(1, -1)
        
        brand, color, category, quality = "unable to predict", "Black", "Apparel", "Good"
        
        if image_models_ready:
            if brand_clf:
                dist, ind = brand_clf.kneighbors(features)
                # Increased threshold to 15000 to account for real product complexity vs pure logos
                if dist[0][0] < 15000:
                    brand = str(brand_clf.predict(features)[0])
            
            if color_clf:
                color = str(color_clf.predict(features)[0])
            if cat_clf:
                category = str(cat_clf.predict(features)[0])
            if quality_clf:
                quality = str(quality_clf.predict(features)[0])
                
        return jsonify({
            "brand": brand,
            "color": color,
            "category": category,
            "quality": quality
        })
    except Exception as e:
        print(f"Analysis error: {e}")
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)
