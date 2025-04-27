from flask import Flask, render_template, request, redirect, flash, session
import pyodbc
import pandas as pd
import os
import plotly.graph_objs as go
import plotly
import json
import joblib
from sklearn.preprocessing import LabelEncoder
users = {}  # username -> {password, email}

app = Flask(__name__)
app.secret_key = 'flashing'  # can be any random string
UPLOAD_FOLDER = './uploads'
ALLOWED_EXTENSIONS = {'csv'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# Database connection settings
server = 'retail-bd.database.windows.net'
database = 'retail_db'
username = 'jagan'
password = 'retailbd1!'
driver = '{ODBC Driver 17 for SQL Server}'

def get_db_connection():
    conn = pyodbc.connect(
        f'DRIVER={driver};SERVER={server};DATABASE={database};UID={username};PWD={password};Encrypt=yes;TrustServerCertificate=no'
    )
    return conn
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        user = users.get(username)
        if user and user['password'] == password:
            session['username'] = username
            flash('Logged in successfully!', 'success')
            return redirect('/')
        else:
            flash('Invalid username or password', 'danger')

    return render_template('login.html')
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        email = request.form['email']

        if username in users:
            flash('Username already exists!', 'danger')
        else:
            users[username] = {'password': password, 'email': email}
            flash('Registration successful! Please log in.', 'success')
            return redirect('/login')

    return render_template('register.html')
@app.route('/logout')
def logout():
    session.pop('username', None)
    flash('Logged out successfully!', 'success')
    return redirect('/login')

@app.route('/')
def home():
    if 'username' not in session:
        return redirect('/login')
    return render_template('home.html', username=session['username'])
@app.route('/upload_options')
def upload_options():
    return render_template('upload_options.html')
# Sample Data Pull for HSHD_NUM = 10
@app.route('/sample')
def sample_data():
    conn = get_db_connection()
    query = """
    SELECT 
        t.hshd_num, 
        t.basket_num, 
        t.date, 
        t.product_num, 
        p.department, 
        p.commodity,
        h.loyalty_flag,
        h.age_range,
        h.marital_status,
        h.income_range,
        h.homeowner_desc,
        h.hshd_composition,
        h.hshd_size,
        h.children
    FROM dbo.transactions t
    JOIN dbo.products p ON t.product_num = p.product_num
    JOIN dbo.households h ON t.hshd_num = h.hshd_num
    WHERE t.hshd_num = 10
    ORDER BY t.hshd_num, t.basket_num, t.date, t.product_num, p.department, p.commodity
    """
    df = pd.read_sql(query, conn)
    conn.close()

    return render_template('sample.html', tables=[df.to_html(classes='data', index=False)], titles=df.columns.values)

# Interactive search page
@app.route('/search', methods=['GET', 'POST'])
def search():
    if request.method == 'POST':
        
        hshd_num = request.form.get('hshd_num')
        conn = get_db_connection()
        query = """
        SELECT 
            t.hshd_num, 
            t.basket_num, 
            t.date, 
            t.product_num, 
            p.department, 
            p.commodity,
            h.loyalty_flag,
            h.age_range,
            h.marital_status,
            h.income_range,
            h.homeowner_desc,
            h.hshd_composition,
            h.hshd_size,
            h.children
        FROM dbo.transactions t
        JOIN dbo.products p ON t.product_num = p.product_num
        JOIN dbo.households h ON t.hshd_num = h.hshd_num
        WHERE t.hshd_num = ?
        ORDER BY t.hshd_num, t.basket_num, t.date, t.product_num, p.department, p.commodity
        """
        df = pd.read_sql(query, conn, params=[hshd_num])
        conn.close()

        return render_template('result.html', tables=[df.to_html(classes='data', index=False)], titles=df.columns.values)
    return render_template('search.html')
@app.route('/insert_transaction', methods=['GET', 'POST'])
def insert_transaction():
    if request.method == 'POST':
        hshd_num = request.form['hshd_num']
        basket_num = request.form['basket_num']
        date = request.form['date']
        product_num = request.form['product_num']
        spend = request.form['spend']
        units = request.form['units']
        store_region = request.form['store_region']
        week_num = request.form['week_num']
        year = request.form['year']

        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO dbo.transactions (hshd_num, basket_num, date, product_num, spend, units, store_region, week_num, year)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, 
        hshd_num, basket_num, date, product_num, spend, units, store_region, week_num, year)

        conn.commit()
        cursor.close()
        conn.close()
        flash('✅ Transaction inserted successfully!')
        return redirect('/')  # Go back to Home after inserting
    return render_template('insert_transaction.html')

@app.route('/insert_household', methods=['GET', 'POST'])
def insert_household():
    if request.method == 'POST':
        hshd_num = request.form['hshd_num']
        loyalty_flag = request.form['loyalty_flag']
        age_range = request.form['age_range']
        marital_status = request.form['marital_status']
        income_range = request.form['income_range']
        homeowner_desc = request.form['homeowner_desc']
        hshd_composition = request.form['hshd_composition']
        hshd_size = request.form['hshd_size']
        children = request.form['children']

        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO dbo.households (hshd_num, loyalty_flag, age_range, marital_status, income_range, homeowner_desc, hshd_composition, hshd_size, children)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        hshd_num, loyalty_flag, age_range, marital_status, income_range, homeowner_desc, hshd_composition, hshd_size, children)

        conn.commit()
        cursor.close()
        conn.close()
        flash('✅ Transaction inserted successfully!')
        return redirect('/upload_options')  # After inserting, go back to upload options
    return render_template('insert_household.html')

@app.route('/insert_product', methods=['GET', 'POST'])
def insert_product():
    if request.method == 'POST':
        product_num = request.form['product_num']
        department = request.form['department']
        commodity = request.form['commodity']
        brand_type = request.form['brand_type']
        natural_organic_flag = request.form['natural_organic_flag']

        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO dbo.products (product_num, department, commodity, brand_type, natural_organic_flag)
            VALUES (?, ?, ?, ?, ?)
        """,
        product_num, department, commodity, brand_type, natural_organic_flag)

        conn.commit()
        cursor.close()
        conn.close()
        flash('✅ Transaction inserted successfully!')
        return redirect('/upload_options')  # After inserting, go back to upload options
    return render_template('insert_product.html')

@app.route('/load_csv', methods=['GET', 'POST'])
def load_csv():
    if request.method == 'POST':
        files = request.files.getlist('csv_files')

        if not files or files == []:
            flash('❌ No files uploaded.')
            return redirect('/load_csv')

        if not os.path.exists(app.config['UPLOAD_FOLDER']):
            os.makedirs(app.config['UPLOAD_FOLDER'])

        conn = get_db_connection()
        cursor = conn.cursor()

        for file in files:
            filename = file.filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            try:
                # Read and strip spaces
                df = pd.read_csv(filepath)
                df.columns = df.columns.str.strip()
                df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

                table = None

                # Detect table based on CSV columns
                if set(['HSHD_NUM', 'L', 'AGE_RANGE', 'MARITAL', 'INCOME_RANGE', 'HOMEOWNER', 'HSHD_COMPOSITION', 'HH_SIZE', 'CHILDREN']).issubset(set(df.columns)):
                    table = 'households'
                    
                    # Preprocess
                    df['HSHD_NUM'] = pd.to_numeric(df['HSHD_NUM'], errors='coerce').fillna(0).astype(int)
                    text_fields = ['L', 'AGE_RANGE', 'MARITAL', 'INCOME_RANGE', 'HOMEOWNER', 'HSHD_COMPOSITION', 'HH_SIZE', 'CHILDREN']
                    for col in text_fields:
                        df[col] = df[col].fillna('UNKNOWN')

                    # Insert
                    for _, row in df.iterrows():
                        cursor.execute("""
                            INSERT INTO households (
                                hshd_num, loyalty_flag, age_range, marital_status, 
                                income_range, homeowner_desc, hshd_composition, 
                                hshd_size, children
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        row['HSHD_NUM'],
                        row['L'],
                        row['AGE_RANGE'],
                        row['MARITAL'],
                        row['INCOME_RANGE'],
                        row['HOMEOWNER'],
                        row['HSHD_COMPOSITION'],
                        row['HH_SIZE'],
                        row['CHILDREN']
                        )

                    flash(f"✅ Households data from {filename} loaded successfully!")

                elif set(['PRODUCT_NUM', 'DEPARTMENT', 'COMMODITY', 'BRAND_TY', 'NATURAL_ORGANIC_FLAG']).issubset(set(df.columns)):
                    table = 'products'
                    
                    # Preprocess
                    df['PRODUCT_NUM'] = pd.to_numeric(df['PRODUCT_NUM'], errors='coerce').fillna(0).astype(int)
                    text_fields_products = ['DEPARTMENT', 'COMMODITY', 'BRAND_TY', 'NATURAL_ORGANIC_FLAG']
                    for col in text_fields_products:
                        df[col] = df[col].fillna('UNKNOWN')

                    # Insert
                    for _, row in df.iterrows():
                        cursor.execute("""
                            INSERT INTO products (
                                product_num, department, commodity, brand_type, natural_organic_flag
                            ) VALUES (?, ?, ?, ?, ?)
                        """,
                        row['PRODUCT_NUM'],
                        row['DEPARTMENT'],
                        row['COMMODITY'],
                        row['BRAND_TY'],
                        row['NATURAL_ORGANIC_FLAG']
                        )

                    flash(f"✅ Products data from {filename} loaded successfully!")

                elif set(['HSHD_NUM', 'BASKET_NUM', 'PURCHASE_', 'PRODUCT_NUM', 'SPEND', 'UNITS', 'STORE_R', 'WEEK_NUM', 'YEAR']).issubset(set(df.columns)):
                    table = 'transactions'

                    # Preprocess
                    for col in ['HSHD_NUM', 'BASKET_NUM', 'PRODUCT_NUM', 'SPEND', 'UNITS', 'WEEK_NUM', 'YEAR']:
                        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

                    # Insert
                    for _, row in df.iterrows():
                        cursor.execute("""
                            INSERT INTO transactions (
                                hshd_num, basket_num, date, product_num,
                                spend, units, store_region, week_num, year
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        row['HSHD_NUM'],
                        row['BASKET_NUM'],
                        row['PURCHASE_'],
                        row['PRODUCT_NUM'],
                        row['SPEND'],
                        row['UNITS'],
                        row['STORE_R'],
                        row['WEEK_NUM'],
                        row['YEAR']
                        )

                    flash(f"✅ Transactions data from {filename} loaded successfully!")

                else:
                    flash(f"❌ File {filename} does not match any table schema. Skipped.")

            except Exception as e:
                flash(f"❌ Error loading {filename}: {str(e)}")
                continue

        conn.commit()
        cursor.close()
        conn.close()

        return redirect('/upload_options')

    return render_template('load_csv.html')




@app.route('/dashboard', methods=['GET', 'POST'])
def dashboard():
    conn = get_db_connection()

    selected_year = request.args.get('year')
    selected_income = request.args.get('income')

    # Dropdown Options
    years = pd.read_sql("SELECT DISTINCT year FROM transactions ORDER BY year", conn)['year'].tolist()
    incomes = pd.read_sql("SELECT DISTINCT income_range FROM households WHERE income_range IS NOT NULL", conn)['income_range'].tolist()

    # Filters
    year_filter = f"AND t.year = {selected_year}" if selected_year else ""
    income_filter = f"AND h.income_range = '{selected_income}'" if selected_income else ""

    # 1. Demographics and Engagement
    demo_query = f"""
    SELECT income_range, AVG(spend) AS avg_spend
    FROM transactions t
    JOIN households h ON t.hshd_num = h.hshd_num
    WHERE 1=1 {year_filter}
    GROUP BY income_range
    """
    demo_df = pd.read_sql(demo_query, conn)

    fig1 = go.Figure([go.Bar(x=demo_df['income_range'], y=demo_df['avg_spend'])])
    fig1.update_layout(title='Income Range vs Average Spend')

    # 2. Engagement Over Time
    engagement_query = f"""
        SELECT t.year, AVG(t.spend) AS avg_spend
        FROM transactions t
        JOIN households h ON t.hshd_num = h.hshd_num
        WHERE 1=1 {income_filter}
        GROUP BY t.year
        ORDER BY t.year
        """
    engagement_df = pd.read_sql(engagement_query, conn)

    fig2 = go.Figure([go.Scatter(x=engagement_df['year'], y=engagement_df['avg_spend'], mode='lines+markers')])
    fig2.update_layout(title='Average Spend Over Years')

    # 3. Basket Analysis
    basket_query = f"""
    SELECT product_num, COUNT(basket_num) AS count
    FROM transactions t
    WHERE 1=1 {year_filter}
    GROUP BY product_num
    ORDER BY count DESC
    OFFSET 0 ROWS FETCH NEXT 10 ROWS ONLY
    """
    basket_df = pd.read_sql(basket_query, conn)
    print(basket_df)
    fig3 = go.Figure([go.Bar(x=basket_df['product_num'].astype(str), y=basket_df['count'])])
    print("test1",fig3)
    fig3.update_layout(title='Top Products in Baskets')
    print("test2",fig3)
    # 4. Seasonal Trends
    seasonal_query = f"""
    SELECT week_num, SUM(spend) AS total_spend
    FROM transactions t
    WHERE 1=1 {year_filter}
    GROUP BY week_num
    ORDER BY week_num
    """
    seasonal_df = pd.read_sql(seasonal_query, conn)

    fig4 = go.Figure([go.Scatter(x=seasonal_df['week_num'], y=seasonal_df['total_spend'], mode='lines+markers')])
    fig4.update_layout(title='Seasonal Trends (Spend by Week)',
                       xaxis_title='Product Number',
                       yaxis_title='Number of Baskets',
                       xaxis_type='category')

    # 5. Brand Preferences
    brand_query = f"""
    SELECT brand_type, COUNT(*) AS cnt
    FROM transactions t
    JOIN products p ON t.product_num = p.product_num
    WHERE 1=1 {year_filter}
    GROUP BY brand_type
    """
    brand_df = pd.read_sql(brand_query, conn)
    fig5 = go.Figure([go.Bar(x=brand_df['brand_type'], y=brand_df['cnt'])])
    fig5.update_layout(title='Brand Preferences')

    conn.close()

    graphs = {
    'fig1': fig1,
    'fig2': fig2,
    'fig3': fig3,
    'fig4': fig4,
    'fig5': fig5
    }

    return render_template('dashboard.html', graphs=json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder),
                           years=years,
                           incomes=incomes,
                           selected_year=selected_year,
                           selected_income=selected_income)



model_clv = joblib.load("model/gradient_boosting_clv.pkl")
model_basket = joblib.load("model/random_forest_basket.pkl")
model_churn = joblib.load("model/logistic_regression_churn.pkl")

encoder = LabelEncoder()
store_regions = ['CENTRAL', 'EAST', 'SOUTH', 'WEST']
encoder.fit(store_regions)

@app.route('/clv_predict', methods=['GET', 'POST'])
def clv_predict():
    result = None
    if request.method == 'POST':
        spend = float(request.form['Spend'])
        units = int(request.form['Units'])
        income_range = int(request.form['Income_range'])
        hshd_size = int(request.form['Hshd_size'])
        children = int(request.form['Children'])

        df = pd.DataFrame({
            'Spend': [spend],
            'Units': [units],
            'Hshd_size': [hshd_size],
            'Children': [children],
            'Income_range': [income_range]
        })

        prediction = model_clv.predict(df)
        result = round(prediction[0], 2)

    return render_template('clv_predict.html', result=result)


@app.route('/basket_predict', methods=['GET', 'POST'])
def basket_predict():
    result = None
    if request.method == 'POST':
        spend = float(request.form['Spend'])
        units = int(request.form['Units'])
        income_range = int(request.form['Income_range'])
        hshd_size = int(request.form['Hshd_size'])
        children = int(request.form['Children'])
        store_region = request.form['Store_region']

        df = pd.DataFrame({
            'Spend': [spend],
            'Units': [units],
            'Hshd_size': [hshd_size],
            'Children': [children],
            'Income_range': [income_range],
            'Store_region_encoded': encoder.transform([store_region])
        })

        prediction = model_basket.predict(df)
        result = prediction[0]

    return render_template('basket_predict.html', result=result)


@app.route('/churn_predict', methods=['GET', 'POST'])
def churn_predict():
    result = None
    if request.method == 'POST':
        spend = float(request.form['Spend'])
        units = int(request.form['Units'])
        income_range = int(request.form['Income_range'])
        hshd_size = int(request.form['Hshd_size'])
        children = int(request.form['Children'])

        df = pd.DataFrame({
            'Spend': [spend],
            'Units': [units],
            'Hshd_size': [hshd_size],
            'Children': [children],
            'Income_range': [income_range]
        })

        prediction = model_churn.predict(df)
        result = prediction[0]

    return render_template('churn_predict.html', result=result)


@app.route('/model_predictions')
def model_predictions():
    return render_template('model_predictions.html')

if __name__ == '__main__':
    app.run(debug=True)