import psycopg2


class verification:
    def __init__(self):
        self.conn=psycopg2.connect(
            host="localhost",
            database='call_agent',
            user="postgres",
            password='root',
            port="5432"
)

        self.cursor=self.conn.cursor()

    def verification(self,connection_number,customer_nic,customer_name):
        query=""""
                SELECT * FROM customers
                WHERE connection_number = %s AND customer_nic = %s AND customer_name = %s;
        """
        self.cursor.execute(query,(connection_number,customer_nic,customer_name))
        result=self.cursor.fetchone()
        
        return bool(result)
    
        if result:
            return True, 
        else:
            return False
    
        while True:
            conn_num=input()
            name=input()
            nic=input()
        
    
        valid,message = verification(name,nic,conn_num)
    
        if valid:
            return
    
    def close(self):
        self.cursor.close()
        self.conn.close()
    
    


