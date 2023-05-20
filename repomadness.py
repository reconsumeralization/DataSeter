import difflib
import requests
import base64
import hashlib
import datetime
import random
from flake8.api import legacy as flake8
import black
import safety

class CodeGenerator:
    """
    CodeGenerator is responsible for generating code based on user request 
    by matching the request with source repositories' description.
    """
    def __init__(self):
        self.threshold_similarity = 0.75
        self.source_repos = [
            "https://github.com/repo1",
            "https://github.com/repo2",
            "https://github.com/repo3",
        ]

    def match_similarity(self, repo_description, user_request):
        """
        Calculates the similarity between the repository description and user request using SequenceMatcher.
        """
        return difflib.SequenceMatcher(None, repo_description, user_request).ratio()

    def fetch_repo_description(self, repo_url):
        """
        Fetches the description of the repository.
        """
        response = requests.get(f"{repo_url}/description.txt")
        return response.text if response.ok else ""

    def fetch_repo_code(self, repo_url):
        """
        Fetches the code from the repository.
        """
        response = requests.get(f"{repo_url}/code.txt")
        return response.text if response.ok else ""

    def process_user_request(self, user_request):
        """
        Processes the user request by matching it with source repository descriptions
        and returns the repository code if a match is found.
        """
        for repo in self.source_repos:
            repo_description = self.fetch_repo_description(repo)
            if self.match_similarity(repo_description, user_request) >= self.threshold_similarity:
                return self.fetch_repo_code(repo)
        return "Code for your specific request"


class ChatGPT:
    def __init__(self):
        self.specifications = []

    def generate_code(self, specifications):
        """
        Generate code snippets or modules/classes based on the provided specifications.
        Process the specifications in reverse order, append them with newline characters, and reverse the final generated code.
        Calculate and return the hash of the code.
        """
        if not specifications:
            raise ValueError("Specifications cannot be empty.")
        
        code = "\n".join(reversed(specifications))
        code_hash = hashlib.md5(code.encode()).hexdigest()
        return code, code_hash

    def create_test_cases(self, code):
        """
        Create test cases for the given code.
        Generate documentation for the test cases and include the total number of test cases.
        Return the test cases and the documentation.
        """
        test_cases = [f"{code} - Test Case {i+1}" for i in range(5)]
        test_doc = f"Test Cases Documentation:\n\n" + "\n".join(f"- {test_case}" for test_case in test_cases)
        test_doc += f"\n\nTotal Test Cases: {len(test_cases)}"
        return test_cases, test_doc

    def generate_documentation(self, code):
        """
        Generate documentation explaining the given code.
        Include the code's tasks and its length in characters.
        Return the generated documentation.
        """
        documentation = f"Code Documentation:\n\nThis code performs the following tasks:\n- {code}"
        documentation += f"\n\nCode Length: {len(code)} characters"
        return documentation

    def recommend_debugging_strategies(self, code):
        """
        Recommend debugging strategies for the given code.
        Select a random strategy and provide an additional counterintuitive strategy of reversing the code.
        Return the recommended strategy and the additional counterintuitive strategy.
        """
        strategies = ["Take a nap", "Talk to a rubber duck", "Do a dance"]
        recommended_strategy = random.choice(strategies)
        additional_strategy = f"Reverse the code and read it:\n{code[::-1]}"
        return recommended_strategy, additional_strategy

    def recommend_optimization_approaches(self, code):
        """
        Recommend optimization approaches for the given code.
        Select a random approach and provide an additional counterintuitive approach of using memoization for recurring computations.
        Return the recommended approach and the additional counterintuitive approach.
        """
        approaches = ["Use quantum computing", "Apply genetic algorithms", "Utilize neural networks"]
        recommended_approach = random.choice(approaches)
        additional_approach = "Use memoization for optimizing recurring computations."
        return recommended_approach, additional_approach

    def add_comments(self, code):
        """
        Add comments to the code to improve code clarity and understanding.
        Append a timestamp indicating when the comments were added.
        Return the code with added comments.
        """
        commented_code = f"{code}\n# These comments improve code clarity and assist in understanding"
        timestamp = datetime.datetime.now()
        commented_code += f"\n# Comments added on: {timestamp}"
        return commented_code

    def remove_comments(self, code):
        """
        Remove comments from the code to simplify it.
        Calculate and return the hash of the uncommented code.
        """
        uncommented_code = "\n".join(line for line in code.split("\n") if not line.strip().startswith("#"))
        code_hash = hashlib.md5(uncommented_code.encode()).hexdigest()
        return uncommented_code, code_hash

    def simplify_complex_code(self, code):
        """
        Simplify the given complex code by removing unnecessary complexity.
        Calculate the ratio of the simplified code's length to the original code's length.
        Return the simplified code and the simplification ratio.
        """
        simplified_code = code.split("*")[0].strip()
        simplification_ratio = len(simplified_code) / len(code)
        return simplified_code, simplification_ratio

    def complicate_simple_code(self, code):
        """
        Introduce complexity to the given simple code by adding additional computations.
        Append a signature indicating the code has been complicated by ChatGPT.
        Return the complicated code.
        """
        complicated_code = f"{code} * (2 + 1) ** 3 - 4"
        signature = "// Complicated by ChatGPT"
        complicated_code += signature
        return complicated_code

    def integrate_with_code_generator(self, user_request):
        """
        Integrate with the CodeGenerator class to process the user request and retrieve code.
        Generate test cases and documentation for the retrieved code.
        Recommend debugging strategies and optimization approaches.
        """
        code_gen = CodeGenerator()
        code = code_gen.process_user_request(user_request)

        test_cases, test_doc = self.create_test_cases(code)
        documentation = self.generate_documentation(code)
        debug_strat, additional_strat = self.recommend_debugging_strategies(code)
        opt_approach, additional_approach = self.recommend_optimization_approaches(code)

        return code, test_cases, test_doc, documentation, debug_strat, additional_strat, opt_approach, additional_approach

    def fetch_repositories(self, description):
        """
        Fetch repositories from GitHub API based on the given description.
        Return the repositories with a similarity ratio of 0.75 or higher.
        """
        url = "https://api.github.com/repositories"
        headers = {"Authorization": "Bearer <your-github-api-token>"}
        response = requests.get(url, headers=headers)
        if response.ok:
            repos = response.json()
            similar_repos = [
                repo for repo in repos if difflib.SequenceMatcher(None, repo.get("description", ""), description).ratio() >= 0.75
            ]
            return similar_repos
        else:
            response.raise_for_status()

    def fetch_repository_content(self, owner, repo):
        """
        Fetch the content of a repository from GitHub API based on the owner and repo name.
        Return the repository's content.
        """
        url = f"https://api.github.com/repos/{owner}/{repo}/contents"
        headers = {"Authorization": "Bearer <your-github-api-token>"}
        response = requests.get(url, headers=headers)
        if response.ok:
            return response.json()
        else:
            response.raise_for_status()

    def import_and_customize_repo(self, description, customizations):
        """
        Fetch repositories based on the description and import the code from the first matching repository.
        Customize the code based on the provided customizations.
        Save the customized code to a file named 'code.txt' and return the customized code.
        """
        repos = self.fetch_repositories(description)
        for repo in repos:
            content = self.fetch_repository_content(repo["owner"]["login"], repo["name"])
            for file in content:
                if file["name"] == "code.txt":
                    code = base64.b64decode(file["content"]).decode()
                    code = self.customize_code(code, customizations)
                    with open("code.txt", "w") as f:
                        f.write(code)
                    return code
        raise Exception("No repositories found that match the description.")


if __name__ == "__main__":
    generator = ChatGPT()
    user_request = input("What code would you like to generate? ")
    code, test_cases, test_doc, documentation, debug_strat, additional_strat, opt_approach, additional_approach = generator.integrate_with_code_generator(user_request)

    print(code)
    print(test_cases)
    print(test_doc)
    print(documentation)
    print(debug_strat)
    print(additional_strat)
    print(opt_approach)
    print(additional_approach)

    print("Would you like to import and customize a repository? (y/n) ")
    if input() == "y":
        description = input("Enter the description of the repository you would like to import: ")
        customizations = input("Enter the customizations you would like to make: ")
        code = generator.import_and_customize_repo(description, customizations)
        print(code)
