"""
Data Loader Module
Parses voters.sql dump file and extracts voter records
"""
import re
import pandas as pd
from typing import List, Dict, Any


def parse_sql_dump(file_path: str) -> List[Dict[str, Any]]:
    """
    Parse the voters.sql dump file and extract voter records.
    
    Args:
        file_path: Path to the SQL dump file
        
    Returns:
        List of voter dictionaries
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Define column names based on the CREATE TABLE statement
    columns = [
        'id', 'serial_bn', 'serial', 'name', 'name_normalized',
        'voter_id_bn', 'voter_id', 'father_name', 'father_name_normalized',
        'mother_name', 'occupation', 'date_of_birth', 'address',
        'voter_area_no_bn', 'voter_area_no', 'union', 'ward_bn', 'ward',
        'gender', 'created_at', 'updated_at', 'phonetic_name', 'phonetic_father_name'
    ]
    
    voters = []
    
    # Find all INSERT statements and extract values
    # Pattern to match each row of values in the INSERT statement
    insert_pattern = r'\((\d+),\s*\'([^\']*)\',\s*\'([^\']*)\',\s*\'([^\']*)\',\s*\'([^\']*)\',\s*(NULL|\'[^\']*\'),\s*(NULL|\'[^\']*\'),\s*\'([^\']*)\',\s*\'([^\']*)\',\s*\'([^\']*)\',\s*(NULL|\'[^\']*\'),\s*\'([^\']*)\',\s*\'([^\']*)\',\s*\'([^\']*)\',\s*\'([^\']*)\',\s*\'([^\']*)\',\s*\'([^\']*)\',\s*\'([^\']*)\',\s*\'([^\']*)\',\s*\'([^\']*)\',\s*\'([^\']*)\',\s*\'([^\']*)\',\s*\'([^\']*)\'\)'
    
    matches = re.findall(insert_pattern, content)
    
    for match in matches:
        voter = {}
        for i, col in enumerate(columns):
            if i < len(match):
                value = match[i]
                # Clean up NULL values and quotes
                if value == 'NULL':
                    voter[col] = None
                elif value.startswith("'") and value.endswith("'"):
                    voter[col] = value[1:-1]
                else:
                    voter[col] = value
            else:
                voter[col] = None
        voters.append(voter)
    
    # If regex didn't work well, try alternative parsing
    if len(voters) == 0:
        voters = parse_sql_alternative(content, columns)
    
    return voters


def parse_sql_alternative(content: str, columns: List[str]) -> List[Dict[str, Any]]:
    """
    Alternative parsing method using line-by-line approach.
    """
    voters = []
    
    # Find the INSERT statement section
    lines = content.split('\n')
    in_values = False
    current_values = ""
    
    for line in lines:
        if 'INSERT INTO' in line and 'voters' in line:
            in_values = True
            continue
        
        if in_values:
            current_values += line
    
    # Parse individual records
    # Split by "),\n(" pattern to get individual records
    records = re.split(r'\),\s*\n?\(', current_values)
    
    for record in records:
        # Clean up the record
        record = record.strip()
        if record.startswith('('):
            record = record[1:]
        if record.endswith(');'):
            record = record[:-2]
        if record.endswith(')'):
            record = record[:-1]
        
        if not record:
            continue
            
        # Parse values from the record
        values = parse_record_values(record)
        
        if len(values) >= len(columns):
            voter = {}
            for i, col in enumerate(columns):
                value = values[i] if i < len(values) else None
                if value == 'NULL' or value is None:
                    voter[col] = None
                else:
                    voter[col] = str(value).strip("'")
            voters.append(voter)
    
    return voters


def parse_record_values(record: str) -> List[str]:
    """
    Parse comma-separated values, handling quoted strings properly.
    """
    values = []
    current = ""
    in_quotes = False
    
    for char in record:
        if char == "'" and not in_quotes:
            in_quotes = True
            current += char
        elif char == "'" and in_quotes:
            in_quotes = False
            current += char
        elif char == ',' and not in_quotes:
            values.append(current.strip())
            current = ""
        else:
            current += char
    
    if current:
        values.append(current.strip())
    
    # Clean values
    cleaned = []
    for v in values:
        v = v.strip()
        if v.startswith("'") and v.endswith("'"):
            v = v[1:-1]
        elif v == 'NULL':
            v = None
        cleaned.append(v)
    
    return cleaned


def create_voter_documents(voters: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Create searchable text documents from voter records.
    
    Args:
        voters: List of voter dictionaries
        
    Returns:
        List of documents with text content and metadata
    """
    documents = []
    
    for voter in voters:
        # Create a rich text representation for semantic search
        text_parts = []
        
        # Name variations for better search
        if voter.get('name'):
            text_parts.append(f"নাম (Name): {voter['name']}")
        if voter.get('phonetic_name'):
            text_parts.append(f"Phonetic Name: {voter['phonetic_name']}")
        
        # Father's name
        if voter.get('father_name'):
            text_parts.append(f"পিতার নাম (Father's Name): {voter['father_name']}")
        if voter.get('phonetic_father_name'):
            text_parts.append(f"Phonetic Father: {voter['phonetic_father_name']}")
        
        # Mother's name
        if voter.get('mother_name'):
            text_parts.append(f"মাতার নাম (Mother's Name): {voter['mother_name']}")
        
        # Occupation
        if voter.get('occupation'):
            text_parts.append(f"পেশা (Occupation): {voter['occupation']}")
        
        # Date of birth
        if voter.get('date_of_birth'):
            text_parts.append(f"জন্ম তারিখ (Date of Birth): {voter['date_of_birth']}")
        
        # Address
        if voter.get('address'):
            text_parts.append(f"ঠিকানা (Address): {voter['address']}")
        
        # Ward and Union
        if voter.get('ward'):
            text_parts.append(f"ওয়ার্ড নং (Ward No): {voter['ward']}")
        if voter.get('ward_bn'):
            text_parts.append(f"ওয়ার্ড (বাংলা): {voter['ward_bn']}")
        if voter.get('union'):
            text_parts.append(f"ইউনিয়ন (Union): {voter['union']}")
        
        # Gender
        if voter.get('gender'):
            text_parts.append(f"লিঙ্গ (Gender): {voter['gender']}")
        
        # Serial number
        if voter.get('serial'):
            text_parts.append(f"ক্রমিক নং (Serial): {voter['serial']}")
        
        # Create the document
        doc = {
            'id': str(voter.get('id', '')),
            'content': '\n'.join(text_parts),
            'metadata': {
                'id': str(voter.get('id', '')),
                'name': voter.get('name', ''),
                'father_name': voter.get('father_name', ''),
                'mother_name': voter.get('mother_name', ''),
                'occupation': voter.get('occupation', ''),
                'ward': voter.get('ward', ''),
                'union': voter.get('union', ''),
                'gender': voter.get('gender', ''),
                'date_of_birth': voter.get('date_of_birth', ''),
                'address': voter.get('address', ''),
                'serial': voter.get('serial', '')
            }
        }
        documents.append(doc)
    
    return documents


def load_voters_from_sql(file_path: str) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Main function to load voters from SQL dump and create documents.
    
    Args:
        file_path: Path to the SQL dump file
        
    Returns:
        Tuple of (raw voters list, documents list)
    """
    print(f"Loading voters from {file_path}...")
    voters = parse_sql_dump(file_path)
    print(f"Parsed {len(voters)} voter records")
    
    documents = create_voter_documents(voters)
    print(f"Created {len(documents)} searchable documents")
    
    return voters, documents


def get_statistics(voters: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calculate statistics from voter data.
    """
    df = pd.DataFrame(voters)
    
    stats = {
        'total_voters': len(voters),
        'by_occupation': df['occupation'].value_counts().to_dict() if 'occupation' in df.columns else {},
        'by_ward': df['ward'].value_counts().to_dict() if 'ward' in df.columns else {},
        'by_gender': df['gender'].value_counts().to_dict() if 'gender' in df.columns else {},
        'unions': df['union'].unique().tolist() if 'union' in df.columns else []
    }
    
    return stats


if __name__ == "__main__":
    # Test the data loader
    from config import SQL_DUMP_PATH
    
    voters, documents = load_voters_from_sql(SQL_DUMP_PATH)
    stats = get_statistics(voters)
    
    print("\n--- Statistics ---")
    print(f"Total voters: {stats['total_voters']}")
    print(f"\nBy Occupation: {stats['by_occupation']}")
    print(f"\nBy Ward: {stats['by_ward']}")
    print(f"\nBy Gender: {stats['by_gender']}")
    
    if documents:
        print("\n--- Sample Document ---")
        print(documents[0]['content'])
