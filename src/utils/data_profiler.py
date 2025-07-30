import json
from typing import Dict, List, Any, Optional
import pandas as pd
from sqlalchemy import inspect
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
import pickle
from collections import defaultdict

logger = logging.getLogger(__name__)

class DataProfiler:
    def __init__(self, db_handler):
        """Initialize data profiler with database handler"""
        self.db_handler = db_handler
        
    def _get_column_stats(self, data: List[Dict[str, Any]], column: str) -> Dict[str, Any]:
        """Calculate statistics for a single column"""
        values = [row[column] for row in data if column in row]
        if not values:
            return {"count": 0, "unique_count": 0, "null_count": 0}
            
        df = pd.Series(values)
        stats = {
            "count": len(values),
            "unique_count": df.nunique(),
            "null_count": df.isna().sum(),
            "sample_values": df.sample(min(5, len(df))).tolist()
        }
        
        # Add type-specific statistics
        if pd.api.types.is_numeric_dtype(df):
            stats.update({
                "min": float(df.min()) if not pd.isna(df.min()) else None,
                "max": float(df.max()) if not pd.isna(df.max()) else None,
                "mean": float(df.mean()) if not pd.isna(df.mean()) else None,
                "std": float(df.std()) if not pd.isna(df.std()) else None
            })
        elif pd.api.types.is_string_dtype(df):
            stats.update({
                "min_length": int(df.str.len().min()),
                "max_length": int(df.str.len().max()),
                "avg_length": float(df.str.len().mean())
            })
            
        return stats
    
    def profile_database(self, 
                        db_name: str, 
                        table_name: str, 
                        sample_size: int = 1000) -> Dict[str, Any]:
        """Generate comprehensive profile for a database table"""
        db = self.db_handler.connections[db_name]
        inspector = inspect(db.engine)
        
        # Get schema information
        columns = inspector.get_columns(table_name)
        schema_info = {
            "column_count": len(columns),
            "columns": {
                col['name']: {
                    "type": str(col['type']),
                    "nullable": col.get('nullable', True),
                    "default": str(col.get('default', None))
                } for col in columns
            }
        }
        
        # Get sample data
        query = f"SELECT * FROM {table_name} LIMIT {sample_size}"
        sample_data = self.db_handler.connections[db_name].execute_query(query)
        
        # Calculate statistics for each column
        column_stats = {
            col['name']: self._get_column_stats(sample_data, col['name'])
            for col in columns
        }
        
        return {
            "table_name": table_name,
            "schema": schema_info,
            "statistics": {
                "row_count": len(sample_data),
                "columns": column_stats
            },
            "sample_data": sample_data[:5] if sample_data else []
        }
    
    def export_profile(self, 
                      profile_data: Dict[str, Any], 
                      output_dir: str = "data/profiles",
                      filename: Optional[str] = None) -> str:
        """Export profile data to JSON file"""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"profile_{profile_data['table_name']}_{timestamp}.json"
            
        output_path = Path(output_dir) / filename
        
        with open(output_path, 'w') as f:
            json.dump(profile_data, f, indent=2, default=str)
            
        logger.info(f"Exported profile to {output_path}")
        return str(output_path)
    
    def compare_profiles(self, 
                        source_profile: Dict[str, Any], 
                        target_profile: Dict[str, Any]) -> Dict[str, Any]:
        """Compare source and target profiles to identify differences"""
        comparison = {
            "schema_differences": {
                "source_only_columns": [],
                "target_only_columns": [],
                "type_mismatches": []
            },
            "data_distribution": {
                "value_ranges": {},
                "null_ratios": {}
            }
        }
        
        # Compare schemas
        source_cols = set(source_profile["schema"]["columns"].keys())
        target_cols = set(target_profile["schema"]["columns"].keys())
        
        comparison["schema_differences"]["source_only_columns"] = list(source_cols - target_cols)
        comparison["schema_differences"]["target_only_columns"] = list(target_cols - source_cols)
        
        # Compare column types and distributions
        for col in source_cols & target_cols:
            source_type = source_profile["schema"]["columns"][col]["type"]
            target_type = target_profile["schema"]["columns"][col]["type"]
            
            if source_type != target_type:
                comparison["schema_differences"]["type_mismatches"].append({
                    "column": col,
                    "source_type": source_type,
                    "target_type": target_type
                })
            
            # Compare value distributions
            source_stats = source_profile["statistics"]["columns"][col]
            target_stats = target_profile["statistics"]["columns"][col]
            
            comparison["data_distribution"]["value_ranges"][col] = {
                "source": {k: source_stats[k] for k in ["min", "max", "mean", "std"] 
                          if k in source_stats},
                "target": {k: target_stats[k] for k in ["min", "max", "mean", "std"]
                          if k in target_stats}
            }
            
            # Compare null ratios
            source_null_ratio = source_stats["null_count"] / source_stats["count"]
            target_null_ratio = target_stats["null_count"] / target_stats["count"]
            comparison["data_distribution"]["null_ratios"][col] = {
                "source": source_null_ratio,
                "target": target_null_ratio
            }
            
        return comparison

class EnhancedDataProfiler(DataProfiler):
    def __init__(self, db_handler, embedding_handler=None):
        super().__init__(db_handler)
        self.embedding_handler = embedding_handler
    
    def profile_source_independently(self, source_name: str, table_name: str) -> dict:
        """Profile source table without knowing target mapping"""
        profile = self.profile_database(source_name, table_name)
        profile.update({
            "source_name": source_name,
            "data_quality": self._assess_data_quality(source_name, table_name),
            "patterns": self._detect_field_patterns(source_name, table_name)
        })
        return profile
    
    def profile_target_independently(self, target_name: str, table_name: str) -> dict:
        """Profile target table independently"""
        profile = self.profile_database(target_name, table_name)
        profile.update({
            "target_name": target_name,
            "data_quality": self._assess_data_quality(target_name, table_name),
            "patterns": self._detect_field_patterns(target_name, table_name)
        })
        return profile
    
    def _assess_data_quality(self, db_name: str, table_name: str) -> dict:
        """Assess data quality metrics"""
        profile = self.profile_database(db_name, table_name)
        
        quality_metrics = {
            "null_ratio": 0.0,
            "duplicate_ratio": 0.0,
            "data_completeness": 0.0,
            "data_consistency": 0.0
        }
        
        total_rows = profile["statistics"]["row_count"]
        if total_rows == 0:
            return quality_metrics
        
        # Calculate null ratio
        total_nulls = sum(
            col_stats["null_count"] 
            for col_stats in profile["statistics"]["columns"].values()
        )
        total_cells = total_rows * len(profile["statistics"]["columns"])
        quality_metrics["null_ratio"] = total_nulls / total_cells if total_cells > 0 else 0.0
        
        # Calculate data completeness
        quality_metrics["data_completeness"] = 1.0 - quality_metrics["null_ratio"]
        
        return quality_metrics
    
    def _detect_field_patterns(self, db_name: str, table_name: str) -> dict:
        """Detect patterns in field names and data"""
        schema = self.db_handler.get_source_schema(db_name, table_name)
        
        patterns = {
            "naming_conventions": self._analyze_naming_conventions(schema),
            "field_categories": self._categorize_fields(schema),
            "data_patterns": self._analyze_data_patterns(db_name, table_name)
        }
        return patterns
    
    def _analyze_naming_conventions(self, schema):
        """Analyze naming conventions used in the schema"""
        conventions = {
            "snake_case": 0,
            "camel_case": 0,
            "pascal_case": 0,
            "uppercase": 0
        }
        
        for field in schema.keys():
            if field.isupper():
                conventions["uppercase"] += 1
            elif "_" in field and field.islower():
                conventions["snake_case"] += 1
            elif field[0].isupper() and not "_" in field:
                conventions["pascal_case"] += 1
            elif not field[0].isupper() and any(c.isupper() for c in field[1:]):
                conventions["camel_case"] += 1
            else:
                # Default to snake_case for simple lowercase names
                conventions["snake_case"] += 1
        
        return conventions
    
    def _categorize_fields(self, schema: dict) -> dict:
        """Categorize fields based on naming patterns"""
        categories = defaultdict(list)
        
        for field_name in schema.keys():
            field_lower = field_name.lower()
            
            if any(word in field_lower for word in ['id', 'identifier']):
                categories["identifiers"].append(field_name)
            elif any(word in field_lower for word in ['name', 'title']):
                categories["names"].append(field_name)
            elif any(word in field_lower for word in ['date', 'time']):
                categories["dates"].append(field_name)
            elif any(word in field_lower for word in ['email', 'phone', 'contact']):
                categories["contact_info"].append(field_name)
            elif any(word in field_lower for word in ['address', 'location']):
                categories["addresses"].append(field_name)
            else:
                categories["other"].append(field_name)
        
        return dict(categories)
    
    def _analyze_data_patterns(self, db_name: str, table_name: str) -> dict:
        """Analyze data patterns in the table"""
        # This is a simplified version - in practice, you'd analyze actual data
        return {
            "pattern_types": ["email", "phone", "date", "numeric"],
            "pattern_confidence": 0.8
        }
    
    def analyze_potential_mappings(self, 
                                 source_profile: dict, 
                                 target_profile: dict) -> dict:
        """Analyze potential mappings before actual mapping"""
        
        analysis = {
            "potential_mappings": {},
            "field_similarities": {},
            "data_compatibility_matrix": {},
            "mapping_recommendations": {}
        }
        
        source_fields = source_profile["statistics"]["columns"].keys()
        target_fields = target_profile["statistics"]["columns"].keys()
        
        # Calculate field similarities
        for source_field in source_fields:
            analysis["field_similarities"][source_field] = {}
            for target_field in target_fields:
                similarity = self._calculate_field_similarity(
                    source_field, 
                    target_field,
                    source_profile["statistics"]["columns"][source_field],
                    target_profile["statistics"]["columns"][target_field]
                )
                analysis["field_similarities"][source_field][target_field] = similarity
        
        # Generate potential mappings
        analysis["potential_mappings"] = self._generate_potential_mappings(
            analysis["field_similarities"]
        )
        
        return analysis
#calculate the potential mapping
    def _calculate_field_similarity(self, 
                                  source_field: str, 
                                  target_field: str,
                                  source_stats: dict,
                                  target_stats: dict) -> float:
        """Calculate similarity between fields using multiple factors"""
        
        # Simple string similarity (in practice, use embeddings)
        semantic_sim = self._get_string_similarity(source_field, target_field)
        
        # Data type similarity
        type_sim = self._get_type_similarity(source_stats.get('type'), target_stats.get('type'))
        
        # Data pattern similarity
        pattern_sim = self._get_pattern_similarity(source_stats, target_stats)
        
        # Weighted combination
        similarity = (0.5 * semantic_sim + 0.3 * type_sim + 0.2 * pattern_sim)
        
        return similarity
    
    def _get_string_similarity(self, str1: str, str2: str) -> float:
        """Calculate string similarity"""
        str1_lower = str1.lower().replace('_', ' ')
        str2_lower = str2.lower().replace('_', ' ')
        
        # Simple Jaccard similarity
        set1 = set(str1_lower.split())
        set2 = set(str2_lower.split())
        
        if not set1 and not set2:
            return 1.0
        if not set1 or not set2:
            return 0.0
        
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        return intersection / union if union > 0 else 0.0
    
    def _get_type_similarity(self, type1: str, type2: str) -> float:
        """Calculate type similarity"""
        if not type1 or not type2:
            return 0.0
        
        type1_lower = type1.lower()
        type2_lower = type2.lower()
        
        # Exact match
        if type1_lower == type2_lower:
            return 1.0
        
        # Similar types
        if any(t in type1_lower for t in ['varchar', 'text', 'string']) and \
           any(t in type2_lower for t in ['varchar', 'text', 'string']):
            return 0.8
        
        if any(t in type1_lower for t in ['int', 'integer', 'number']) and \
           any(t in type2_lower for t in ['int', 'integer', 'number']):
            return 0.8
        
        return 0.0
    
    def _get_pattern_similarity(self, source_stats: dict, target_stats: dict) -> float:
        """Calculate pattern similarity"""
        # Simple pattern comparison
        source_has_numeric = 'min' in source_stats and 'max' in source_stats
        target_has_numeric = 'min' in target_stats and 'max' in target_stats
        
        if source_has_numeric == target_has_numeric:
            return 0.8
        return 0.2
    
    def _generate_potential_mappings(self, similarities: dict) -> dict:
        """Generate potential mappings based on similarities"""
        potential_mappings = {}
        
        for source_field, target_similarities in similarities.items():
            # Find best matches for each source field
            sorted_targets = sorted(
                target_similarities.items(),
                key=lambda x: x[1],
                reverse=True
            )
            
            potential_mappings[source_field] = {
                "top_matches": sorted_targets[:3],  # Top 3 matches
                "best_match": sorted_targets[0] if sorted_targets else None,
                "confidence": sorted_targets[0][1] if sorted_targets else 0.0
            }
        
        return potential_mappings
    
    def profile_for_mapping(self, 
                           source_name: str, 
                           target_name: str,
                           source_table: str,
                           target_table: str) -> dict:
        """Profile data with mapping considerations"""
        
        # Independent profiling
        source_profile = self.profile_source_independently(source_name, source_table)
        target_profile = self.profile_target_independently(target_name, target_table)
        
        # Pre-mapping analysis
        pre_mapping_analysis = self.analyze_potential_mappings(source_profile, target_profile)
        
        # Data quality assessment
        data_quality = self.assess_overall_data_quality(source_profile, target_profile)
        
        return {
            "source_profile": source_profile,
            "target_profile": target_profile,
            "pre_mapping_analysis": pre_mapping_analysis,
            "data_quality_assessment": data_quality,
            "mapping_readiness": self.assess_mapping_readiness(source_profile, target_profile)
        }
    
    def assess_overall_data_quality(self, source_profile: dict, target_profile: dict) -> dict:
        """Assess overall data quality"""
        source_quality = source_profile.get("data_quality", {})
        target_quality = target_profile.get("data_quality", {})
        
        return {
            "source_quality_score": 1.0 - source_quality.get("null_ratio", 0.0),
            "target_quality_score": 1.0 - target_quality.get("null_ratio", 0.0),
            "overall_quality_score": (1.0 - source_quality.get("null_ratio", 0.0) + 
                                    1.0 - target_quality.get("null_ratio", 0.0)) / 2
        }
    
    def assess_mapping_readiness(self, source_profile: dict, target_profile: dict) -> dict:
        """Assess if data is ready for mapping"""
        
        readiness = {
            "overall_score": 0.0,
            "issues": [],
            "recommendations": []
        }
        
        # Check schema complexity
        source_complexity = len(source_profile["statistics"]["columns"])
        target_complexity = len(target_profile["statistics"]["columns"])
        
        if source_complexity > 50:
            readiness["issues"].append("Source schema is very complex (>50 fields)")
            readiness["recommendations"].append("Consider breaking into smaller mappings")
        
        # Check data quality
        source_quality = source_profile.get("data_quality", {})
        target_quality = target_profile.get("data_quality", {})
        
        if source_quality.get("null_ratio", 0) > 0.3:
            readiness["issues"].append("High null ratio in source data")
            readiness["recommendations"].append("Investigate data quality issues")
        
        # Calculate overall readiness score
        readiness["overall_score"] = self._calculate_readiness_score(
            source_profile, target_profile
        )
        
        return readiness
    
    def _calculate_readiness_score(self, source_profile: dict, target_profile: dict) -> float:
        """Calculate overall readiness score for mapping"""
        score = 1.0
        
        # Deduct points for issues
        source_quality = source_profile.get("data_quality", {})
        target_quality = target_profile.get("data_quality", {})
        
        # High null ratios
        if source_quality.get("null_ratio", 0) > 0.5:
            score -= 0.3
        elif source_quality.get("null_ratio", 0) > 0.2:
            score -= 0.1
        
        # Schema complexity
        source_fields = len(source_profile["statistics"]["columns"])
        target_fields = len(target_profile["statistics"]["columns"])
        
        if source_fields > 100 or target_fields > 100:
            score -= 0.2
        
        return max(0.0, score)

class PostMappingProfiler:
    def __init__(self, db_handler, schema_mapper):
        self.db_handler = db_handler
        self.schema_mapper = schema_mapper
    
    def create_mapping_aware_comparison(self, 
                                      source_profile: dict, 
                                      target_profile: dict,
                                      mapping: dict) -> dict:
        """Create comparison based on actual mappings"""
        
        comparison = {
            "mapped_fields": {},
            "unmapped_source_fields": [],
            "unmapped_target_fields": [],
            "data_compatibility": {},
            "transformation_insights": {}
        }
        
        # Analyze mapped fields
        for source_field, mapping_info in mapping.items():
            target_field = mapping_info['target_field']
            confidence = mapping_info['confidence']
            
            comparison["mapped_fields"][source_field] = {
                "target_field": target_field,
                "confidence": confidence,
                "source_stats": source_profile["statistics"]["columns"].get(source_field, {}),
                "target_stats": target_profile["statistics"]["columns"].get(target_field, {}),
                "compatibility_score": self._calculate_compatibility(
                    source_profile["statistics"]["columns"].get(source_field, {}),
                    target_profile["statistics"]["columns"].get(target_field, {})
                )
            }
        
        # Find unmapped fields
        source_fields = set(source_profile["statistics"]["columns"].keys())
        target_fields = set(target_profile["statistics"]["columns"].keys())
        mapped_source_fields = set(mapping.keys())
        mapped_target_fields = set(mapping_info['target_field'] for mapping_info in mapping.values())
        
        comparison["unmapped_source_fields"] = list(source_fields - mapped_source_fields)
        comparison["unmapped_target_fields"] = list(target_fields - mapped_target_fields)
        
        return comparison
    
    def _calculate_compatibility(self, source_stats: dict, target_stats: dict) -> float:
        """Calculate compatibility score between source and target fields"""
        score = 0.0
        
        # Type compatibility
        if self._are_types_compatible(source_stats.get('type'), target_stats.get('type')):
            score += 0.3
        
        # Value range compatibility
        if self._are_ranges_compatible(source_stats, target_stats):
            score += 0.3
        
        # Data distribution compatibility
        if self._are_distributions_compatible(source_stats, target_stats):
            score += 0.4
        
        return score
    
    def _are_types_compatible(self, source_type: str, target_type: str) -> bool:
        """Check if types are compatible"""
        if not source_type or not target_type:
            return False
        
        source_lower = source_type.lower()
        target_lower = target_type.lower()
        
        # String types
        if any(t in source_lower for t in ['varchar', 'text', 'string']) and \
           any(t in target_lower for t in ['varchar', 'text', 'string']):
            return True
        
        # Numeric types
        if any(t in source_lower for t in ['int', 'integer', 'number']) and \
           any(t in target_lower for t in ['int', 'integer', 'number']):
            return True
        
        return source_lower == target_lower
    
    def _are_ranges_compatible(self, source_stats: dict, target_stats: dict) -> bool:
        """Check if value ranges are compatible"""
        if 'min' not in source_stats or 'max' not in source_stats:
            return True  # Can't determine, assume compatible
        
        if 'min' not in target_stats or 'max' not in target_stats:
            return True  # Can't determine, assume compatible
        
        # Check if source range fits within target range
        source_min = source_stats['min']
        source_max = source_stats['max']
        target_min = target_stats['min']
        target_max = target_stats['max']
        
        if source_min is None or source_max is None or target_min is None or target_max is None:
            return True
        
        return source_min >= target_min and source_max <= target_max
    
    def _are_distributions_compatible(self, source_stats: dict, target_stats: dict) -> bool:
        """Check if data distributions are compatible"""
        # Simple check - if both have similar null ratios
        source_null_ratio = source_stats.get('null_count', 0) / source_stats.get('count', 1)
        target_null_ratio = target_stats.get('null_count', 0) / target_stats.get('count', 1)
        
        return abs(source_null_ratio - target_null_ratio) < 0.2

    def compare_profiles_enhanced(self, 
                                 source_profile: Dict[str, Any], 
                                 target_profile: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced comparison that works even when column names are different"""
        comparison = {
            "schema_differences": {
                "source_only_columns": [],
                "target_only_columns": [],
                "type_mismatches": [],
                "potential_mappings": {}
            },
            "data_distribution": {
                "value_ranges": {},
                "null_ratios": {},
                "compatibility_analysis": {}
            },
            "similarity_analysis": {
                "overall_similarity_score": 0.0,
                "field_similarities": {},
                "best_matches": {}
            }
        }
        
        # Get column sets
        source_cols = set(source_profile["schema"]["columns"].keys())
        target_cols = set(target_profile["schema"]["columns"].keys())
        
        # Basic exact name comparison
        common_cols = source_cols & target_cols
        comparison["schema_differences"]["source_only_columns"] = list(source_cols - target_cols)
        comparison["schema_differences"]["target_only_columns"] = list(target_cols - source_cols)
        
        # Enhanced analysis for different column names
        if not common_cols:  # No exact name matches
            comparison["schema_differences"]["potential_mappings"] = self._find_potential_mappings_by_similarity(
                source_profile, target_profile
            )
            
            # Analyze data compatibility for potential matches
            comparison["data_distribution"]["compatibility_analysis"] = self._analyze_data_compatibility(
                source_profile, target_profile, 
                comparison["schema_differences"]["potential_mappings"]
            )
            
            # Calculate overall similarity score
            comparison["similarity_analysis"]["overall_similarity_score"] = self._calculate_overall_similarity(
                comparison["schema_differences"]["potential_mappings"]
            )
            
        else:  # Some exact name matches exist
            # Handle exact matches
            for col in common_cols:
                source_type = source_profile["schema"]["columns"][col]["type"]
                target_type = target_profile["schema"]["columns"][col]["type"]
                
                if source_type != target_type:
                    comparison["schema_differences"]["type_mismatches"].append({
                        "column": col,
                        "source_type": source_type,
                        "target_type": target_type
                    })
                
                # Compare value distributions for exact matches
                source_stats = source_profile["statistics"]["columns"][col]
                target_stats = target_profile["statistics"]["columns"][col]
                
                comparison["data_distribution"]["value_ranges"][col] = {
                    "source": {k: source_stats[k] for k in ["min", "max", "mean", "std"] 
                              if k in source_stats},
                    "target": {k: target_stats[k] for k in ["min", "max", "mean", "std"]
                              if k in target_stats}
                }
                
                # Compare null ratios
                source_null_ratio = source_stats["null_count"] / source_stats["count"]
                target_null_ratio = target_stats["null_count"] / target_stats["count"]
                comparison["data_distribution"]["null_ratios"][col] = {
                    "source": source_null_ratio,
                    "target": target_null_ratio
                }
            
            # Also find potential mappings for non-exact matches
            remaining_source = source_cols - common_cols
            remaining_target = target_cols - common_cols
            
            if remaining_source and remaining_target:
                comparison["schema_differences"]["potential_mappings"] = self._find_potential_mappings_by_similarity(
                    source_profile, target_profile, remaining_source, remaining_target
                )
        
        return comparison
    
    def _find_potential_mappings_by_similarity(self, 
                                             source_profile: Dict[str, Any], 
                                             target_profile: Dict[str, Any],
                                             source_cols: set = None,
                                             target_cols: set = None) -> Dict[str, Any]:
        """Find potential mappings using similarity analysis"""
        if source_cols is None:
            source_cols = set(source_profile["schema"]["columns"].keys())
        if target_cols is None:
            target_cols = set(target_profile["schema"]["columns"].keys())
        
        potential_mappings = {}
        
        for source_col in source_cols:
            best_matches = []
            for target_col in target_cols:
                similarity = self._calculate_field_similarity(
                    source_col, target_col,
                    source_profile["statistics"]["columns"][source_col],
                    target_profile["statistics"]["columns"][target_col]
                )
                best_matches.append((target_col, similarity))
            
            # Sort by similarity and take top 3
            best_matches.sort(key=lambda x: x[1], reverse=True)
            potential_mappings[source_col] = {
                "top_matches": best_matches[:3],
                "best_match": best_matches[0] if best_matches else None,
                "confidence": best_matches[0][1] if best_matches else 0.0
            }
        
        return potential_mappings
    
    def _analyze_data_compatibility(self, 
                                  source_profile: Dict[str, Any], 
                                  target_profile: Dict[str, Any],
                                  potential_mappings: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze data compatibility for potential mappings"""
        compatibility_analysis = {}
        
        for source_col, mapping_info in potential_mappings.items():
            best_match = mapping_info.get("best_match")
            if best_match:
                target_col = best_match[0]
                source_stats = source_profile["statistics"]["columns"][source_col]
                target_stats = target_profile["statistics"]["columns"][target_col]
                
                compatibility_analysis[f"{source_col} -> {target_col}"] = {
                    "type_compatibility": self._are_types_compatible(
                        source_profile["schema"]["columns"][source_col]["type"],
                        target_profile["schema"]["columns"][target_col]["type"]
                    ),
                    "range_compatibility": self._are_ranges_compatible(source_stats, target_stats),
                    "distribution_compatibility": self._are_distributions_compatible(source_stats, target_stats),
                    "overall_compatibility_score": self._calculate_compatibility_score(
                        source_stats, target_stats,
                        source_profile["schema"]["columns"][source_col]["type"],
                        target_profile["schema"]["columns"][target_col]["type"]
                    )
                }
        
        return compatibility_analysis
    
    def _calculate_overall_similarity(self, potential_mappings: Dict[str, Any]) -> float:
        """Calculate overall similarity score for the schema comparison"""
        if not potential_mappings:
            return 0.0
        
        total_confidence = sum(mapping["confidence"] for mapping in potential_mappings.values())
        return total_confidence / len(potential_mappings)
    
    def _calculate_compatibility_score(self, 
                                     source_stats: Dict[str, Any], 
                                     target_stats: Dict[str, Any],
                                     source_type: str, 
                                     target_type: str) -> float:
        """Calculate overall compatibility score between two fields"""
        score = 0.0
        
        # Type compatibility (40% weight)
        if self._are_types_compatible(source_type, target_type):
            score += 0.4
        
        # Range compatibility (30% weight)
        if self._are_ranges_compatible(source_stats, target_stats):
            score += 0.3
        
        # Distribution compatibility (30% weight)
        if self._are_distributions_compatible(source_stats, target_stats):
            score += 0.3
        
        return score

if __name__ == "__main__":
    # Test data profiling
    from ..db.db_handler import MultiSourceDBHandler
    from ..data.mock_data_generator import HealthcareDataGenerator
    import yaml
    
    # Load config
    with open('config/db_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize components
    db_handler = MultiSourceDBHandler({
        name: db_config['connection_string']
        for name, db_config in config['databases'].items()
    })
    
    profiler = EnhancedDataProfiler(db_handler)
    
    # Profile source and target databases
    source1_profile = profiler.profile_database('source1', 'providers')
    source2_profile = profiler.profile_database('source2', 'physicians')
    target_profile = profiler.profile_database('target', 'healthcare_providers')
    
    # Export profiles
    profiler.export_profile(source1_profile)
    profiler.export_profile(source2_profile)
    profiler.export_profile(target_profile)
    
    # Compare profiles
    comparison = profiler.compare_profiles(source1_profile, target_profile)
    profiler.export_profile(
        comparison,
        filename="profile_comparison_source1_target.json"
    ) 