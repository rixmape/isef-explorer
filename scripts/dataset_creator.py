import logging
from pathlib import Path
from typing import Dict, Any

import yaml
import pandas as pd
from bs4 import BeautifulSoup

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_config(config_file: str = "config.yaml") -> Dict[str, Any]:
    """
    Load configuration from a YAML file.

    Args:
        config_file (str): Path to the YAML configuration file.

    Returns:
        Dict[str, Any]: Configuration dictionary.

    Raises:
        FileNotFoundError: If the configuration file is not found.
        yaml.YAMLError: If there's an error parsing the YAML file.
    """
    try:
        with open(config_file, "r") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        logger.error(f"Configuration file not found: {config_file}")
        raise
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML file: {e}")
        raise


def read_html_file(file_path: str) -> BeautifulSoup:
    """
    Read and parse the HTML file.

    Args:
        file_path (str): Path to the HTML file.

    Returns:
        BeautifulSoup: Parsed HTML content.

    Raises:
        FileNotFoundError: If the input file is not found.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return BeautifulSoup(f, "html.parser")
    except FileNotFoundError:
        logger.error(f"Input file not found: {file_path}")
        raise


def extract_data(soup: BeautifulSoup, table_id: str, base_url: str) -> pd.DataFrame:
    """
    Extract project data from the parsed HTML.

    Args:
        soup (BeautifulSoup): Parsed HTML content.
        table_id (str): ID of the table containing the data.
        base_url (str): Base URL for constructing full links.

    Returns:
        pd.DataFrame: DataFrame containing project data.
    """
    table = soup.select_one(f"#{table_id}")
    if not table:
        logger.error(f"Table with id '{table_id}' not found in HTML")
        return pd.DataFrame()

    rows = table.select("tr")
    headers = [header.text.strip() for header in rows[0].select("th")] + ["Link"]

    data = []
    for row in rows[1:]:
        cells = row.select("td")
        project_data = {header: " ".join(cell.text.split()) for header, cell in zip(headers[:-1], cells)}
        project_data["Link"] = base_url + row.select_one("a")["href"]
        data.append(project_data)

    return pd.DataFrame(data)


def main() -> None:
    """
    Main function to orchestrate the ISEF data extraction process.
    """
    try:
        config = load_config()
        logger.info("Starting ISEF data extraction")

        soup = read_html_file(config["input_html_file"])
        df = extract_data(soup, config["table_id"], config["base_url"])

        if df.empty:
            logger.warning("No data extracted from the HTML file")
        else:
            df.to_csv(config["output_csv_file"], index=False)
            logger.info(f"Data successfully written to {config['output_csv_file']}")

        logger.info("ISEF data extraction completed successfully")
    except Exception as e:
        logger.exception(f"An error occurred during the extraction process: {e}")


if __name__ == "__main__":
    main()
