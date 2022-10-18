# Tests for demos

## Test cases

Tests cases are represented as `yaml` file.

## Run tests

To start testing you should run the next script `run_tests.py`.
Running the script with the `-h` option yields the following usage message:
```
usage: run_tests.py [-h] --demo-build-dir DIR --test-data-dir DIR --downloader-cache-dir DIR [--demos DEMO[,DEMO...]]
                    [--scope {base,performance,custom}] [--mo MO.PY] [--devices DEVICES] [--report-file REPORT_FILE] [--log-file LOG_FILE]
                    [--supported-devices SUPPORTED_DEVICES] [--precisions PRECISIONS [PRECISIONS ...]] [--models-dir DIR]

Test script for the demos.

For the tests to work, the test data directory must contain:
* a "BraTS" subdirectory with brain tumor dataset in NIFTI format (see http://medicaldecathlon.com,
  https://drive.google.com/open?id=1A2IU8Sgea1h3fYLpYtFb2v7NYdMjvEhU);
* a "ILSVRC2012_img_val" subdirectory with the ILSVRC2012 dataset;
* a "Image_Retrieval" subdirectory with image retrieval dataset (images, videos) (see https://github.com/19900531/test)
  and list of images (see https://github.com/openvinotoolkit/training_extensions/blob/089de2f/misc/tensorflow_toolkit/image_retrieval/data/gallery/gallery.txt)
* a "msasl" subdirectory with the MS-ASL dataset (https://www.microsoft.com/en-us/research/project/ms-asl/)
* a file how_are_you_doing.wav from (https://storage.openvinotoolkit.org/data/test_data/)
* a file stream_8_high.mp4 from https://storage.openvinotoolkit.org/data/test_data/videos/smartlab/stream_8_high.mp4
* a file stream_8_top.mp4 from https://storage.openvinotoolkit.org/data/test_data/videos/smartlab/stream_8_top.mp4

optional arguments:
  -h, --help            show this help message and exit
  --demo-build-dir DIR  directory with demo binaries
  --test-data-dir DIR   directory with test data
  --downloader-cache-dir DIR
                        directory to use as the cache for the model downloader
  --demos DEMO[,DEMO...]
                        list of demos to run tests for (by default, every demo is tested). For testing demos of specific implementation
                        pass one (or more) of the next values: cpp, cpp_gapi, python.
  --scope {base,performance,custom}
                        The scenario for testing demos.
  --mo MO.PY            Model Optimizer entry point script
  --devices DEVICES     list of devices to test
  --report-file REPORT_FILE
                        path to report file
  --log-file LOG_FILE   path to log file
  --supported-devices SUPPORTED_DEVICES
                        paths to Markdown files with supported devices for each model
  --precisions PRECISIONS [PRECISIONS ...]
                        IR precisions for all models. By default, models are tested in FP16, FP16-INT8 precisions
  --models-dir DIR      directory with pre-converted models (IRs)
```

Example: 
```
python3 run_tests.py --demo-build-dir <path_to_binaries> \
                     --test-data-dir <path_to_folder_with_datasets> \
                     --downloader-cache-dir <path_to_cashe_folder>
```