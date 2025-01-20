import sqlite3
import yfinance as yf
def create_database(db_name):
    """
    Create a database for stock market data
    """
    conn = sqlite3.connect(db_name)
    return conn
def fetch_real_time_data(symbol):
    '''
    Fetches real-time stock data for a given stock symbol.
    Inputs:
        symbol(str): A string of stock symbol
     Returns:
        DataFrame: A pandas DataFrame of stock data
    '''
    stock = yf.Ticker(symbol)
    stock_data = stock.history(period='1d')
    return stock_data

def create_table(conn):
    """
    Create a table for stock market data
    """
    cursor = conn.cursor()
    cursor.execute('''
    CREATE TABLE 
    IF NOT EXISTS stock_companies (id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,industry TEXT NOT NULL
    )
    ''')
    conn.commit()
    return cursor
def recreate_table_if_needed(conn):
    '''
    Check if the table schema matches expectations, and fix it if it doesn't.
    '''
    cursor = conn.cursor()
    # Check if the table exists
    cursor.execute('''
    SELECT name FROM 
    sqlite_master 
    WHERE type='table' 
    AND name='stock_companies'
    ''')
    table_exists = cursor.fetchone()

    if table_exists:
    # Get the table schema
        cursor.execute('''
        PRAGMA table_info(stock_companies)
        ''')
        columns = cursor.fetchall()
        column_names = [col[1] for col in columns]
        # Check if the table schema matches expectations
        if set(column_names) != {'id', 'name', 'industry'}:
            print("Table schema does not match expectations. Recreating the table...")
            # Drop the table
            cursor.execute('''
            DROP TABLE stock_companies
            ''')
            create_table(conn)
        else:
            print("Table schema matches expectations.")
    else:
        # If the table does not exist, create it
        create_table(conn)

        print("Created the table successfully!")



def insert_companies(conn, stock_companies):
    """
    Insert companies into the tables

    """
    cursor = conn.cursor()
    cursor.executemany('''
    INSERT INTO stock_companies (name, industry) VALUES (?,?)
    ''', stock_companies)
    conn.commit()
def main():
    db_name = 'stock_companies.db'
    stock_companies = [
        ('Apple', 'Technology'),
        ('Tesla', 'Automotive'),
        ('Microsoft', 'Technology'),
        ('Amazon', 'Retail'),
        ('Google', 'Technology'),
        ('Facebook', 'Technology'),
        ('Netflix', 'Entertainment'),
        ('Twitter', 'Social Media'),
        ('Shopify', 'Retail'),
        ('Zoom', 'Technology'),
        ('Slack', 'Technology'),
        ('Spotify', 'Entertainment'),
        ('Uber', 'Transportation'),
        ('Lyft', 'Transportation'),
        ('Pinterest', 'Social Media'),
        ('Snapchat', 'Social Media'),
        ('TikTok', 'Social Media'),
        ('Instagram', 'Social Media'),
        ('WhatsApp', 'Social Media'),
        ('LinkedIn', 'Social Media'),
        ('Reddit', 'Social Media'),
        ('Twitch', 'Entertainment'),
        ('Tinder', 'Social Media'),
        ('Bumble', 'Social Media'),
        ('Grindr', 'Social Media'),
        ('Hinge', 'Social Media'),
        ('Match', 'Social Media'),
        ('OkCupid', 'Social Media'),
        ('Plenty of Fish', 'Social Media'),
        ('eHarmony', 'Social Media'),
        ('Zoosk', 'Social Media'),
        ('Christian Mingle', 'Social Media'),
        ('JDate', 'Social Media'),
        ('Farmers Only', 'Social Media'),
        ('Alibaba', 'Retail'),
        ('JD.com', 'Retail'),
        ('Tencent', 'Technology'),
        ('Baidu', 'Technology'),
        ('Weibo', 'Social Media'),
        ('WeChat', 'Social Media'),
        ('QQ', 'Social Media'),
        ('Douyin', 'Social Media'),
        ('Kuaishou', 'Social Media'),
        ('Sina Weibo', 'Social Media'),
        ('YY', 'Social Media'),
        ('VK', 'Social Media'),
        ('Odnoklassniki', 'Social Media'),
        ('MySpace', 'Social Media'),
        ('Friendster', 'Social Media'),
        ('Orkut', 'Social Media'),
        ('Hi5', 'Social Media'),
        ('Bebo', 'Social Media'),
        ('Tagged', 'Social Media'),
        ('Xanga', 'Social Media'),
        ('LiveJournal', 'Social Media'),
        ('Renren', 'Social Media'),
        ('Kaixin', 'Social Media'),
        ('Qzone', 'Social Media'),
        ('Mixi', 'Social Media'),
        ('Cyworld', 'Social Media'),
        ('Naver', 'Social Media'),
        ('Daum', 'Social Media'),
        ('Cyworld', 'Social Media'),
        ('Nate', 'Social Media'),
        ('Me2day', 'Social Media'),
        ('KakaoTalk', 'Social Media'),
        ('Line', 'Social Media'),
        ('Kik', 'Social Media'),
        ('Viber', 'Social Media'),
        ('Telegram', 'Social Media'),
        ('Signal', 'Social Media'),
        ('Threema', 'Social Media'),
        ('Wickr', 'Social Media'),
        ('Snapchat', 'Social Media'),
        ('TikTok', 'Social Media'),
        ('Instagram', 'Social Media'),
        ('WhatsApp', 'Social Media'),
        ('LinkedIn', 'Social Media'),
        ('IBM', 'Technology'),
    ]
    # Create a database
    conn = create_database(db_name)

    # Create a table
    recreate_table_if_needed(conn)

    # Insert companies into the table
    insert_companies(conn, stock_companies)

    # Close the connection
    conn.close()

if __name__ == '__main__':
    main()

