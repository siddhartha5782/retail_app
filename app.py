from flask import Flask, render_template, request
import pyodbc
import pandas as pd

app = Flask(__name__)

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

@app.route('/')
def home():
    return render_template('home.html')

# Sample Data Pull for HSHD_NUM = 10
@app.route('/sample')
def sample_data():
    conn = get_db_connection()
    
    query = """
    SELECT 
        t.hshd_num, t.basket_num, t.product_num, p.department, p.commodity
    FROM transactions t
    JOIN products p ON t.product_num = p.product_num
    WHERE t.hshd_num = 10
    ORDER BY t.hshd_num
    """
    
    df = pd.read_sql(query, conn)
    conn.close()
    
    return render_template('result.html', tables=[df.to_html(classes='data')], titles=df.columns.values)

if __name__ == '__main__':
    app.run(debug=True)
