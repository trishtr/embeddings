from typing import Dict, List, Tuple, Any
from ..embeddings.embedding_handler import EmbeddingHandler
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SchemaMapper:
    def __init__(self, embedding_handler: EmbeddingHandler = None):
        """Initialize schema mapper with embedding handler"""
        self.embedding_handler = embedding_handler or EmbeddingHandler()
        self.mapping_history: List[Dict[str, Any]] = []
        
    def find_field_mappings(self, 
                           source_schema: Dict[str, str],
                           target_schema: Dict[str, str],
                           threshold: float = 0.7) -> List[Tuple[str, str, float]]:
        """Find matching fields between source and target schemas using k-NN"""
        mappings = []
        
        # Use k-NN to find similar fields
        for source_field in source_schema.keys():
            # Find similar fields using k-NN
            similar_fields = self.embedding_handler.find_similar_fields_knn(
                source_field, target_schema
            )
            
            # Filter by threshold and add to mappings
            for target_field, similarity in similar_fields:
                if similarity >= threshold:
                    mappings.append((source_field, target_field, similarity))
        
        # Sort by similarity score
        return sorted(mappings, key=lambda x: x[2], reverse=True)

    def find_field_patterns(self, schema: Dict[str, str]) -> Dict[str, List[str]]:
        """Find patterns in schema using k-NN clustering"""
        return self.embedding_handler.find_field_patterns_knn(schema)

    def find_hierarchical_mappings(self, 
                                 source_schema: Dict[str, str],
                                 target_schema: Dict[str, str]) -> Dict[str, Dict[str, List[Tuple[str, float]]]]:
        """Find hierarchical mappings between schemas"""
        hierarchical_mappings = {}
        
        for source_field in source_schema.keys():
            if '.' in source_field:  # Only process hierarchical fields
                matches = self.embedding_handler.find_hierarchical_matches_knn(
                    source_field, target_schema
                )
                hierarchical_mappings[source_field] = matches
        
        return hierarchical_mappings

    def find_context_aware_mappings(self,
                                  source_schema: Dict[str, str],
                                  target_schema: Dict[str, str],
                                  context: str) -> Dict[str, List[Tuple[str, float]]]:
        """Find context-aware mappings between schemas"""
        context_mappings = {}
        
        for source_field in source_schema.keys():
            matches = self.embedding_handler.find_context_aware_matches_knn(
                source_field, target_schema, context
            )
            context_mappings[source_field] = matches
        
        return context_mappings
    
    def map_schema(self, 
                   source_schema: Dict[str, str],
                   target_schema: Dict[str, str],
                   threshold: float = 0.7) -> Dict[str, Dict[str, Any]]:
        """Map source schema to target schema using embeddings"""
        mappings = self.embedding_handler.find_field_mappings(
            source_schema, target_schema, threshold
        )
        
        result = {}
        for source_field, target_field, confidence in mappings:
            result[source_field] = {
                "target_field": target_field,
                "confidence": confidence,
                "explanation": self.embedding_handler.explain_mapping(
                    source_field, target_field, confidence
                )
            }
            
        # Store mapping history
        self.mapping_history.append({
            "source_schema": source_schema,
            "target_schema": target_schema,
            "mappings": result,
            "threshold": threshold
        })
        
        return result
    
    def transform_data(self, 
                      data: List[Dict[str, Any]], 
                      mapping: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Transform source data to target schema format"""
        transformed_data = []
        
        for record in data:
            transformed_record = {}
            for source_field, mapping_info in mapping.items():
                if source_field in record:
                    target_field = mapping_info["target_field"]
                    transformed_record[target_field] = record[source_field]
            transformed_data.append(transformed_record)
            
        return transformed_data
    
    def export_mapping(self, filepath: str):
        """Export mapping history to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(self.mapping_history, f, indent=2)
    
    def import_mapping(self, filepath: str):
        """Import mapping history from JSON file"""
        with open(filepath, 'r') as f:
            self.mapping_history = json.load(f)
    
    def validate_mapping(self, mapping: Dict[str, Dict[str, Any]], threshold: float = 0.7) -> bool:
        """Validate mapping quality"""
        low_confidence_mappings = [
            (source, info) for source, info in mapping.items()
            if info["confidence"] < threshold
        ]
        
        if low_confidence_mappings:
            logger.warning("Found low confidence mappings:")
            for source, info in low_confidence_mappings:
                logger.warning(f"- {source} -> {info['target_field']} "
                             f"(confidence: {info['confidence']:.2f})")
            return False
        return True

if __name__ == "__main__":
    # Test schema mapping
    from ..data.mock_data_generator import HealthcareDataGenerator
    
    # Generate test data
    generator = HealthcareDataGenerator()
    source_schema = generator.generate_source1_schema()
    target_schema = generator.generate_target_schema()
    source_data = generator.generate_source1_data(5)
    
    # Create and test mapper
    mapper = SchemaMapper()
    mapping = mapper.map_schema(source_schema, target_schema)
    
    print("\nSchema Mapping Results:")
    for source_field, info in mapping.items():
        print(f"\n{info['explanation']}")
    
    # Transform data
    transformed_data = mapper.transform_data(source_data, mapping)
    print("\nTransformed Data Sample:")
    print(json.dumps(transformed_data[0], indent=2)) 