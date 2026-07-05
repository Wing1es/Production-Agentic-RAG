import yaml
from jinja2 import Template
from langsmith import Client
from api.core.config import config

ls_client = Client(api_key=config.LANGSMITH_API_KEY)

def prompt_template_config(yaml_file, prompt_key):

    with open(yaml_file, 'r') as file:
        cfg = yaml.safe_load(file)

    template_content = cfg["prompts"][prompt_key]

    template = Template(template_content)

    return template

def prompt_template_registry(prompt_name):

    template_content = ls_client.pull_prompt(prompt_name).messages[0].prompt.template

    template = Template(template_content)

    return template