from llama_index.llms import GoogleGenerativeAI


class GeminiLLM:
    def __init__(self, model, api_key, max_output_tokens):
        self.model = model
        self.api_key = api_key
        self.max_output_tokens = max_output_tokens
        self.llm = GoogleGenerativeAI(
            model=self.model,
            api_key=self.api_key,
            max_output_tokens=self.max_output_tokens
        )
    
    def answer_with_library_prompt(self, question, context):
        response = self.llm.generate_content(
            question=question,
            context=context
        )
        return response.text
    
    def answer_with_custom_prompt(self, question, context):
        # Tạo prompt theo định dạng mong muốn
        prompt = f"""Hãy đóng vai là một bác sĩ hoặc một chuyên gia tư vấn sức khỏe. Dưới đây là tài liệu tham khảo:
        {context}
        
        Câu hỏi: {question}
        Trả lời chi tiết nhưng ngắn gọn và dễ hiểu dựa trên các tài liệu được tham khảo.
        """
        
        # Gửi prompt vào LLM
        response = self.llm.generate_content(prompt)
        return response.text


