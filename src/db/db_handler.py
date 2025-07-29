from sqlalchemy import create_engine, MetaData, Table, Column, String, inspect
from sqlalchemy.engine import Engine
from typing import Dict, List, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatabaseHandler:
    def __init__(self, connection_string: str):
        """Initialize database connection"""
        self.engine = create_engine(connection_string)
        self.metadata = MetaData()
        
    def get_schema(self, table_name: str) -> Dict[str, str]:
        """Extract schema from existing table"""
        inspector = inspect(self.engine)
        columns = inspector.get_columns(table_name)
        return {col['name']: str(col['type']) for col in columns}
    
    def create_table(self, table_name: str, schema: Dict[str, str]):
        """Create table from schema definition"""
        columns = [Column(name, String) for name, _ in schema.items()]
        Table(table_name, self.metadata, *columns)
        self.metadata.create_all(self.engine)
        logger.info(f"Created table: {table_name}")
    
    def insert_data(self, table_name: str, data: List[Dict[str, Any]]):
        """Insert data into table"""
        if not data:
            return
        
        table = Table(table_name, self.metadata, autoload_with=self.engine)
        with self.engine.connect() as conn:
            conn.execute(table.insert(), data)
            conn.commit()
        logger.info(f"Inserted {len(data)} records into {table_name}")
    
    def execute_query(self, query: str) -> List[Dict[str, Any]]:
        """Execute SQL query and return results"""
        with self.engine.connect() as conn:
            result = conn.execute(query)
            return [dict(row) for row in result]

class MultiSourceDBHandler:
    def __init__(self, connection_strings: Dict[str, str]):
        """Initialize multiple database connections"""
        self.connections: Dict[str, DatabaseHandler] = {
            name: DatabaseHandler(conn_str)
            for name, conn_str in connection_strings.items()
        }
    
    def get_source_schema(self, source_name: str, table_name: str) -> Dict[str, str]:
        """Get schema from source database"""
        return self.connections[source_name].get_schema(table_name)
    
    def get_source_data(self, source_name: str, query: str) -> List[Dict[str, Any]]:
        """Get data from source database"""
        return self.connections[source_name].execute_query(query)
    
    def write_target_data(self, 
                         target_name: str,
                         table_name: str,
                         schema: Dict[str, str],
                         data: List[Dict[str, Any]]):
        """Write data to target database"""
        db = self.connections[target_name]
        db.create_table(table_name, schema)
        db.insert_data(table_name, data)

if __name__ == "__main__":
    # Test database operations
    import os
    from dotenv import load_dotenv
    
    # Load database configurations
    load_dotenv()
    
    # Example connection strings (using SQLite for testing)
    connections = {
        "source1": "sqlite:///source1.db",
        "source2": "sqlite:///source2.db",
        "target": "sqlite:///target.db"
    }
    
    # Initialize handlers
    db_handler = MultiSourceDBHandler(connections)
    
    # Example usage
    from ..data.mock_data_generator import HealthcareDataGenerator
    
    generator = HealthcareDataGenerator()
    source_schema = generator.generate_source1_schema()
    source_data = generator.generate_source1_data(5)
    
    # Create and populate source table
    db_handler.connections["source1"].create_table("providers", source_schema)
    db_handler.connections["source1"].insert_data("providers", source_data)
    
    # Query data
    result = db_handler.get_source_data("source1", "SELECT * FROM providers LIMIT 1")
    print("\nQueried Data Sample:")
    print(result[0]) 