from pydantic import BaseModel, Field
from promptbuilder.prompt_builder import PromptBuilder, schema_to_xml

# Example exactly as requested by the user
class Structure(BaseModel):
    field_i: int = Field(..., description="int field")
    field_s: str = Field(..., description="string field")

def main():
    print("=== User's Requested Example ===")
    print("Input Pydantic Model:")
    print("""
class Structure(BaseModel):
    field_i: int = Field(..., description="int field")
    field_s: str = Field(..., description="string field")
""")
    
    print("Generated XML Schema:")
    xml_schema = schema_to_xml(Structure)
    print(xml_schema)
    
    print("\n=== Using with PromptBuilder ===")
    builder = PromptBuilder()
    builder.text("Extract the following information from the text:\n")
    builder.variable("input_text")
    builder.text("\n\n")
    builder.set_structured_output_xml(Structure, "extracted_data")
    
    prompt = builder.build()
    print("Generated Prompt:")
    print(prompt.prompt_template)
    
    print("\n=== Comparison with JSON Output ===")
    builder_json = PromptBuilder()
    builder_json.text("Extract the following information from the text:\n")
    builder_json.variable("input_text")
    builder_json.text("\n\n")
    builder_json.set_structured_output(Structure, "extracted_data")
    
    prompt_json = builder_json.build()
    print("JSON Version:")
    print(prompt_json.prompt_template)

if __name__ == "__main__":
    main() 