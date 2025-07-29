from faker import Faker
from typing import Dict, List, Any
import random

class HealthcareDataGenerator:
    def __init__(self):
        self.fake = Faker()
        
    def generate_source1_schema(self) -> Dict[str, str]:
        """Schema for source database 1"""
        return {
            "provider_id": "VARCHAR(50)",
            "doctor_name": "VARCHAR(100)",
            "specialization": "VARCHAR(100)",
            "contact_number": "VARCHAR(20)",
            "medical_license": "VARCHAR(50)",
            "practice_location": "VARCHAR(200)"
        }
    
    def generate_source2_schema(self) -> Dict[str, str]:
        """Schema for source database 2 (different naming)"""
        return {
            "physician_identifier": "VARCHAR(50)",
            "full_name": "VARCHAR(100)",
            "medical_specialty": "VARCHAR(100)",
            "phone": "VARCHAR(20)",
            "license_number": "VARCHAR(50)",
            "clinic_address": "VARCHAR(200)"
        }
    
    def generate_target_schema(self) -> Dict[str, str]:
        """Schema for target database"""
        return {
            "healthcare_provider_id": "VARCHAR(50)",
            "provider_name": "VARCHAR(100)",
            "specialty": "VARCHAR(100)",
            "contact_info": "VARCHAR(20)",
            "license_id": "VARCHAR(50)",
            "practice_address": "VARCHAR(200)"
        }

    def generate_source1_data(self, num_records: int) -> List[Dict[str, Any]]:
        """Generate mock data for source 1"""
        data = []
        specializations = ["Cardiology", "Pediatrics", "Neurology", "Oncology", "Family Medicine"]
        
        for _ in range(num_records):
            record = {
                "provider_id": self.fake.uuid4(),
                "doctor_name": self.fake.name(),
                "specialization": random.choice(specializations),
                "contact_number": self.fake.phone_number(),
                "medical_license": f"ML{self.fake.random_number(digits=6)}",
                "practice_location": self.fake.address()
            }
            data.append(record)
        return data

    def generate_source2_data(self, num_records: int) -> List[Dict[str, Any]]:
        """Generate mock data for source 2"""
        data = []
        specialties = ["Cardiac Surgery", "Pediatric Care", "Neurological Medicine", "Cancer Treatment", "General Practice"]
        
        for _ in range(num_records):
            record = {
                "physician_identifier": f"PHY-{self.fake.random_number(digits=6)}",
                "full_name": self.fake.name(),
                "medical_specialty": random.choice(specialties),
                "phone": self.fake.phone_number(),
                "license_number": f"LIC{self.fake.random_number(digits=6)}",
                "clinic_address": self.fake.address()
            }
            data.append(record)
        return data

if __name__ == "__main__":
    # Test data generation
    generator = HealthcareDataGenerator()
    source1_data = generator.generate_source1_data(5)
    source2_data = generator.generate_source2_data(5)
    print("Source 1 Sample:", source1_data[0])
    print("Source 2 Sample:", source2_data[0]) 