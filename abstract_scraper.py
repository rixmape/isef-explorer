import asyncio
import logging
import random
from typing import Any, Dict, Optional

import aiohttp
import pandas as pd
import yaml
from bs4 import BeautifulSoup
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class WebScraper:
    """A class to scrape web content and process CSV files."""

    def __init__(self, config_path: str):
        """
        Initialize the WebScraper with configuration.

        Args:
            config_path (str): Path to the YAML configuration file.
        """
        self.config = self._load_config(config_path)
        self.session: Optional[aiohttp.ClientSession] = None

    @staticmethod
    def _load_config(config_path: str) -> Dict[str, Any]:
        """
        Load configuration from a YAML file.

        Args:
            config_path (str): Path to the YAML configuration file.

        Returns:
            Dict[str, Any]: Configuration dictionary.
        """
        with open(config_path, "r") as config_file:
            return yaml.safe_load(config_file)

    async def _init_session(self):
        """Initialize aiohttp session with connection pooling."""
        if self.session is None:
            self.session = aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=self.config["max_connections"]))

    async def _close_session(self):
        """Close aiohttp session."""
        if self.session:
            await self.session.close()
            self.session = None

    async def scrape_content(self, url: str, css_selector: str) -> str:
        """
        Scrape content from a given URL using the specified CSS selector.

        Args:
            url (str): The URL to scrape.
            css_selector (str): The CSS selector to locate the target element.

        Returns:
            str: The scraped and formatted text.

        Raises:
            ValueError: If the URL is invalid or the content cannot be scraped.
        """
        if not url.startswith(("http://", "https://")):
            raise ValueError(f"Invalid URL: {url}")

        max_retries = self.config["max_retries"]
        base_delay = self.config["retry_delay"]

        for attempt in range(max_retries):
            try:
                async with self.session.get(url, timeout=self.config["request_timeout"]) as response:
                    response.raise_for_status()
                    html = await response.text()

                soup = BeautifulSoup(html, "lxml")
                element = soup.select_one(css_selector)

                if not element:
                    raise ValueError(f"No element found for CSS selector: {css_selector}")

                return self._format_text(element.get_text())
            except (aiohttp.ClientError, ValueError, asyncio.TimeoutError) as e:
                if attempt == max_retries - 1:
                    logger.error(f"Error scraping {url} after {max_retries} attempts: {str(e)}")
                    return ""
                else:
                    delay = base_delay * (2**attempt) + random.uniform(0, 1)
                    logger.warning(f"Attempt {attempt + 1} failed for {url}. Retrying in {delay:.2f} seconds...")
                    await asyncio.sleep(delay)

    @staticmethod
    def _format_text(text: str) -> str:
        """
        Format the scraped text into a single paragraph.

        Args:
            text (str): The raw scraped text.

        Returns:
            str: Formatted text as a single paragraph.
        """
        return " ".join(text.split()).removeprefix("Abstract:")

    async def process_dataframe(self, df: pd.DataFrame, limit: Optional[int] = None) -> pd.DataFrame:
        """
        Process the DataFrame by scraping content for each URL.

        Args:
            df (pd.DataFrame): Input DataFrame with 'Link' column.
            limit (Optional[int]): Maximum number of links to process.

        Returns:
            pd.DataFrame: Updated DataFrame with 'Abstract' column.
        """
        await self._init_session()

        semaphore = asyncio.Semaphore(self.config["concurrent_requests"])

        async def bounded_scrape(url):
            async with semaphore:
                return await self.scrape_content(url, self.config["css_selector"])

        if limit is not None:
            df = df.head(limit)

        tasks = [asyncio.create_task(bounded_scrape(url)) for url in df["Link"]]

        abstracts = []
        for task in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Scraping"):
            abstract = await task
            abstracts.append(abstract)

        df["Abstract"] = abstracts
        return df

    @staticmethod
    def read_csv(file_path: str) -> pd.DataFrame:
        """
        Read the input CSV file.

        Args:
            file_path (str): Path to the input CSV file.

        Returns:
            pd.DataFrame: DataFrame containing the CSV data.

        Raises:
            FileNotFoundError: If the input file does not exist.
            pd.errors.EmptyDataError: If the CSV file is empty.
        """
        try:
            df = pd.read_csv(file_path)
            if "Link" not in df.columns:
                raise ValueError("CSV file must contain a 'Link' column")
            return df
        except FileNotFoundError:
            logger.error(f"Input file not found: {file_path}")
            raise
        except pd.errors.EmptyDataError:
            logger.error(f"Input CSV file is empty: {file_path}")
            raise

    @staticmethod
    def save_csv(df: pd.DataFrame, output_path: str):
        """
        Save the updated DataFrame to a new CSV file.

        Args:
            df (pd.DataFrame): The DataFrame to save.
            output_path (str): Path to the output CSV file.
        """
        df.to_csv(output_path, index=False)
        logger.info(f"Updated CSV saved to: {output_path}")


async def main(input_file: str, output_file: str, config_file: str, limit: Optional[int]):
    """
    Main function to run the web scraping process.

    Args:
        input_file (str): Path to the input CSV file.
        output_file (str): Path to the output CSV file.
        config_file (str): Path to the configuration YAML file.
        limit (Optional[int]): Maximum number of links to process.
    """
    scraper = WebScraper(config_file)

    try:
        df = scraper.read_csv(input_file)
        updated_df = await scraper.process_dataframe(df, limit)
        scraper.save_csv(updated_df, output_file)
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
    finally:
        await scraper._close_session()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Web scraper for CSV data")
    parser.add_argument("input_file", help="Path to the input CSV file")
    parser.add_argument("output_file", help="Path to the output CSV file")
    parser.add_argument("--config", default="config.yaml", help="Path to the configuration file")
    parser.add_argument("--limit", type=int, help="Maximum number of links to scrape")
    args = parser.parse_args()

    asyncio.run(main(args.input_file, args.output_file, args.config, args.limit))
