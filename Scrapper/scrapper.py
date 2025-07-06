from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import os
import json

class CFProblemScraper:
    def __init__(self, chromedriver_path, output_dir="ScrapedProblems"):
        self.base_url = "https://codeforces.com"
        self.chromedriver_path = chromedriver_path
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def _init_driver(self):
        """Initialize Selenium WebDriver with options."""
        options = Options()
        options.add_argument("--window-size=1920,1080")
        service = Service(self.chromedriver_path)
        return webdriver.Chrome(service=service, options=options)

    def _wait_for_problem(self, driver):
        """Wait until the problem content is available on the page."""
        wait = WebDriverWait(driver, 15)
        return wait.until(EC.presence_of_element_located((By.CLASS_NAME, "problem-statement")))

    def scrape_problem(self, pid):
        """Scrape and store problem data by its identifier."""
        url = f"{self.base_url}/problemset/problem/{pid}"
        driver = self._init_driver()

        try:
            driver.get(url)
            self._wait_for_problem(driver)

            soup = BeautifulSoup(driver.page_source, "html.parser")
            problem_div = soup.find("div", class_="problem-statement")

            if not problem_div:
                raise ValueError("Problem content not found.")

            data = self._extract_problem_data(pid, soup, problem_div)
            self._save_text(pid, data["title"], data["description"])
            self._save_metadata(pid, data)

            return data

        except Exception as e:
            print(f"Failed to scrape {pid}: {e}")
            return None
        finally:
            driver.quit()

    def _extract_problem_data(self, pid, soup, problem_div):
        """Extract all required data from the parsed page."""
        title = self._extract_text(problem_div, "title")
        description = self._extract_description(problem_div)
        time_limit = self._extract_text(problem_div, "time-limit")
        memory_limit = self._extract_text(problem_div, "memory-limit")
        tags = [tag.text.strip() for tag in soup.select("span.tag-box")]
        samples = self._extract_samples(problem_div)

        return {
            "title": title,
            "description": description,
            "time_limit": time_limit,
            "memory_limit": memory_limit,
            "tags": tags,
            "samples": samples
        }

    def _extract_text(self, div, class_name):
        """Generic method to extract text from a div by class name."""
        element = div.find("div", class_=class_name)
        return element.text.strip() if element else "N/A"

    def _extract_description(self, div):
        """Assemble a clean problem description."""
        seen = set()
        content = []

        # Add LaTeX spans
        for span in div.find_all("span", class_="math"):
            latex = f"$$ {span.text.strip()} $$"
            if latex not in seen:
                seen.add(latex)
                content.append(latex)

        # Add normal text content
        for tag in div.find_all(["p", "div"]):
            text = tag.text.strip()
            if not text or text.lower().startswith(("input", "output", "time limit", "memory limit")):
                continue
            if text not in seen:
                seen.add(text)
                content.append(text)

        return "\n".join(content)

    def _extract_samples(self, div):
        """Extract all sample input/output pairs."""
        samples = []
        for block in div.find_all("div", class_="sample-test"):
            try:
                input_data = block.find("div", class_="input").find("pre").text.strip()
                output_data = block.find("div", class_="output").find("pre").text.strip()
                samples.append({"input": input_data, "output": output_data})
            except:
                continue
        return samples

    def _save_text(self, pid, title, desc):
        """Save the problem description to a .txt file."""
        filename = os.path.join(self.output_dir, f"{pid.replace('/', '_')}.txt")
        with open(filename, "w", encoding="utf-8") as f:
            f.write(f"Title: {title}\n\n")
            f.write(desc)

    def _save_metadata(self, pid, data):
        """Save metadata (excluding description) to JSON."""
        meta = {k: v for k, v in data.items() if k != "description"}
        filename = os.path.join(self.output_dir, f"{pid.replace('/', '_')}_meta.json")
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

# Usage
if __name__ == "__main__":
    chromedriver = r"C:\Users\prane\Desktop\gdg\chromedriver.exe"
    scraper = CFProblemScraper(chromedriver)

    problems = ['1/A', '1/B', '1/C', '2/A', '2/B', '2/C', '3/A', '3/B', '3/C', '4/A', '4/B', '4/C']

    for pid in problems:
        result = scraper.scrape_problem(pid)
        if result:
            print(f"[✓] Scraped {pid} - {result['title']}")
        else:
            print(f"[✗] Failed to scrape {pid}")
