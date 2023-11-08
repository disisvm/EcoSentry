import mysql.connector
db_config = {
    'host': 'sql10.freemysqlhosting.net',
    'user': 'sql10659817',
    'password': 'Lkrtg7NWLx',
    'database': 'sql10659817'
}

conn = mysql.connector.connect(**db_config)
cursor = conn.cursor()

try:
    cursor.execute("SELECT 1")
    result = cursor.fetchone()
    if result:
        print("Connection is working.")
    else:
        print("Connection might have issues.")

except mysql.connector.Error as e:
    print(f"Error executing the query: {e}")


metadata = {
    'filename': 'example.txt',
    'timestamp': '2023-11-07 18:34:00',
    'user_id': 123
}
#create_query = "create table metadata(filename varchar(100), timestamp datetime, user_id varchar(25))"
#cursor.execute(create_query)

insert_query = "INSERT INTO metadata (filename, timestamp, user_id) VALUES (%s, %s, %s)"
values = (metadata['filename'], metadata['timestamp'], metadata['user_id'])
cursor.execute(insert_query, values)
#print(values)
conn.commit()
conn.close()
