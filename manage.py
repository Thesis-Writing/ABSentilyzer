"""Django's command-line utility for administrative tasks."""

import os
import sys
import subprocess
from nltk.parse.corenlp import CoreNLPDependencyParser
from django.core.management.commands.runserver import Command as runserver
runserver.default_port = "7000"
sys.dont_write_bytecode = True
cwd = os.getcwd()

try:
    path = os.path.join(cwd, 'ensemble_analyzer/apps/public/stanford-corenlp-4.3.1')
    subprocess.Popen('java -mx1g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 8000 -timeout 15000 -quiet', cwd=path, shell =True, stdout=subprocess.PIPE)
    print("From manage.py: {}".format(cwd))
    print("From manage.py: {}".format(path))
except:
    path = os.path.join(cwd, 'ensemble_analyzer/apps/public/stanford-corenlp-4.3.2')
    subprocess.Popen('java -mx1g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 8000 -timeout 15000 -quiet', cwd=path, shell =True, stdout=subprocess.PIPE)

PARSER = CoreNLPDependencyParser(url='http://localhost:8000')

def main():
    """Run administrative tasks."""
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "ensemble_analyzer.settings")
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Are you sure it's installed and "
            "available on your PYTHONPATH environment variable? Did you "
            "forget to activate a virtual environment?"
        ) from exc
    execute_from_command_line(sys.argv)


if __name__ == "__main__":
    main()
