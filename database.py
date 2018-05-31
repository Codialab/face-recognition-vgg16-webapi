import MySQLdb
def database(name_data):
# Open database connection
    db = MySQLdb.connect("localhost","root","","hi" )

# prepare a cursor object using cursor() method
    cursor = db.cursor()
    name_data="'"+name_data + "');"
    # Prepare SQL query to INSERT a record into the database.
    sql = "INSERT INTO `name` (`id`, `name`) VALUES (NULL,"+name_data
    sql2 = "SELECT * FROM name"
    try:
   # Execute the SQL command
       cursor.execute(sql)
   # Commit your changes in the database
       db.commit()
   # Execute the SQL command
       cursor.execute(sql2)
   # Fetch all the rows in a list of lists.
       results = cursor.fetchall()
       for row in results:
           index = row[0]
   # Execute the SQL command    
    except:
   # Rollback in case there is any error
       db.rollback()   
# disconnect from server
    db.close()
    return index

def database_check(name_data):
# Open database connection
    db = MySQLdb.connect("localhost","root","","hi" )
    name_data="'"+name_data + "'"
# prepare a cursor object using cursor() method
    cursor = db.cursor()
    # Prepare SQL query to INSERT a record into the database.
    sql = "SELECT `id` FROM `name` where `name`="+name_data
    try:
   # Execute the SQL command
       cursor.execute(sql)
   # Commit your changes in the database
       db.commit()
   # Fetch all the rows in a list of lists.
       results = cursor.fetchall()
       for row in results:
           index = row[0]
           return index
   # Execute the SQL command    
    except:
   # Rollback in case there is any error
       db.rollback()   
# disconnect from server
    db.close()
    return -1
def database_max():
    # Open database connection
    db = MySQLdb.connect("localhost","root","","hi" )
# prepare a cursor object using cursor() method
    cursor = db.cursor()
    # Prepare SQL query to INSERT a record into the database.
    sql = "SELECT MAX(id) AS id FROM `name`"
    try:
   # Execute the SQL command
       cursor.execute(sql)
   # Commit your changes in the database
       db.commit()
   # Fetch all the rows in a list of lists.
       results = cursor.fetchall()
       for row in results:
           index = row[0]         
   # Execute the SQL command    
    except:
   # Rollback in case there is any error
       db.rollback()   
    return index

def database_get_name(id_data):
# Open database connection
    db = MySQLdb.connect("localhost","root","","hi" )
# prepare a cursor object using cursor() method
    cursor = db.cursor()
    # Prepare SQL query to INSERT a record into the database.
    sql = "SELECT `name` FROM `name` where `id`="+str(id_data+1)
    try:
   # Execute the SQL command
       cursor.execute(sql)
   # Commit your changes in the database
       db.commit()
   # Fetch all the rows in a list of lists.
       results = cursor.fetchall()
       for row in results:
           index = row[0]
           return index
   # Execute the SQL command    
    except:
   # Rollback in case there is any error
       db.rollback()   
# disconnect from server
    db.close()
    return -1
