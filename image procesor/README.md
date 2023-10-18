# DIHM Cell Analyzer

## How to use

1. Clone the repo
   - `gh repo clone varunhc/DIH_demo`
2. Run `app.py` file
   - `python app.py`
3. Open a browser of your choice and go to `127.0.0.1:5000`
4. Upload the image present in the repo at `~/static/uploads`
5. To check if the correct image is uploaded, go to `127.0.0.1:5000/display/<filename>`
6. To view the denoised and segmented image, go to `127.0.0.1:5000/display/result_<filename>`
7. To view the statistics of each type of cell found, go to `127.0.0.1:5000/count_<filename>`
