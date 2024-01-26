# MC Dreamer
## Setup
### Python Environment
Make sure to set up your venv/conda environment with `requirements.txt`. If you do not know what that is, run the following command (assuming you have Python >= 3.11 installed) for Mac/Linux:
```sh
python -m venv .venv
.venv/Scripts/activate
pip install -r requirements.txt
```
or for Windows:
```sh
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```
I would recommend installing the [Python Environment Manager](https://marketplace.visualstudio.com/items?itemName=donjayamanne.python-environment-manager) extension in VSCode, as it will automatically recognize the environment and set it as the default.
### Java Environment
Make sure you have a JDK installed. If you do not, you can download one [here](https://www.oracle.com/java/technologies/downloads/). Make sure to add it to your PATH.

Then, you want to make sure you have Apache Maven set up and [Enklume](https://github.com/Hugobros3/Enklume) added to your local Maven repository. Here is a Maven [download link](https://maven.apache.org/download.cgi) and instructions for [Mac/Linux](https://maven.apache.org/install.html) and [Windows](https://maven.apache.org/guides/getting-started/windows-prerequisites.html).

To add Enklume to your local repository, clone the repository and run `gradlew install` in the root directory. If that does not work, try `gradlew publishToMavenLocal` instead. Then you should have it installed and not get any errors when importing it in the project.

I use IntelliJ IDEA for Java development, so I would recommend using that to open the `java` folder.
### Web Scraping
If you want to do web scraping, there is a little more setup to do. Create a `.env` file in the project directory and fill it out as follows:
```sh
PMC_EMAIL=<Planet Minecraft Email>
PMC_PASSWORD=<Planet Minecraft Password>
```
Since the password used here is in plaintext, I would highly discourage you from using your actual Planet Minecraft account or an actual password; just make a throwaway account for this purpose.

Then make sure `drivers/chromedriver.exe` matches your Chrome version. Currently, it is on version 120. You can check your Chrome version by going to `chrome://settings/help` in Chrome. If it does not match, you can download the correct version [here](https://chromedriver.chromium.org/downloads). Make sure to unzip the file and replace the old `chromedriver.exe` with the new one.

## Heuristic Evaluation
### Overview
The heuristic evaluation is done in `test_heuristic.py`. The heuristics tested are in `core/heuristics.py`; use the `@DisableForTesting` decorator to disable a heuristic for testing. The `test_heuristic.py` file will run the heuristics on a few example inputs in the `test_inputs` folder and output the normalized scores as well as the cosine similarity with the reference scores in `core/heuristics/ref_scores.yaml`.