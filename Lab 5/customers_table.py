import psycopg2

con = psycopg2.connect(database="customers", user="postgres", password="password", host="localhost", port="5432")

print("Database opened successfully")

cur = con.cursor()
#cur.execute('''CREATE TABLE CUSTOMER
      #(ID INT PRIMARY KEY     NOT NULL,
      #NAME           TEXT    NOT NULL,
      #ADDRESS           TEXT    NOT NULL,
      #AGE            INT     NOT NULL,
      #REVIEW        VARCHAR);''')
#cur.execute('''CREATE TABLE CUSTOMER_NOT_INDEXED
      #(ID INT PRIMARY KEY     NOT NULL,
      #NAME           TEXT    NOT NULL,
      #ADDRESS           TEXT    NOT NULL,
      #AGE            INT     NOT NULL,
      #REVIEW        VARCHAR);''')
#print("Table created successfully")

import random
import string

N = 10
def fake_text():
    return ''.join(random.choices(string.ascii_uppercase + " ", k=N))

#for i in range(0,300000):
    #cur.execute(f"INSERT INTO CUSTOMER (ID,Name,Address,age,review) VALUES ('{str(i)}','{fake_text()}','{fake_text()}','{random.randint(1,100)}','{fake_text()}')")
    #con.commit()
    #cur.execute(f"INSERT INTO CUSTOMER_NOT_INDEXED (ID,Name,Address,age,review) VALUES ('{str(i)}','{fake_text()}','{fake_text()}','{random.randint(1,100)}','{fake_text()}')")
    #con.commit()
 
#print("Values inserted successfully")


