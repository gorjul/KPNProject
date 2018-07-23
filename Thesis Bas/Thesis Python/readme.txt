###################
Non standard packages (when using Anaconda) are:

BeautifulSoup
Pandas
Selenium

Note: Selenium requires a webdriver to function. I used geckodriver. For windows machines, gecko and FireFox have to be added to "environment variables" to run properly.

###################
EXPLANATION OF FILES:

SCRIPTS:
-------------------
NRC scrape fin - The NRC job vacancies scraping script. Amount of pages to be scraped is all pages. NOTE: current version doesn't include final CSV due to problem.

ICT scrape - ICTergezocht.nl vacancies scraping script. Number of pages to be scraped can be changed in the 2nd block (standard page = 74 (all))

FD scrape - FD latest news scrape script. The amount of "show more" pages to be scraped can be changed. Note: Large number of iterations will slowdown/crash PC.



CSV: NOTE: Open/import as UTF-8, otherwise some symbols won't be recognized.
-------------------
ict-vacatures-text-all.csv - all scraped info to date for ICTergezocht. Is automatically updated when script is done.

FD-News.csv - all scraped info to date for FD. Is automatically updated when script is done.

###################