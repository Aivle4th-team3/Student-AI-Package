from setuptools import setup, find_packages


with open('README.md', 'r', encoding='utf-8') as fh:
    long_description = fh.read()

setup(
    name='student-ai',
    version='0.4.3',
    description='Package that bundles student AI functions',
    long_description=open('README.md', encoding='utf-8').read(),
    long_description_content_type='text/markdown',
    author='aivle-4th-team3',
    packages=find_packages(),
    package_data={
        'student_ai': ['template_text.json'],
    },
    install_requires=[
        'python-dotenv',
        'langchain',
        'langchain-openai',
        'langchain-google-genai',
        'tiktoken',
        'huggingface-hub',
        'faiss-cpu',
        'langchain-chroma',
        'langchain-pinecone',
    ],
)
