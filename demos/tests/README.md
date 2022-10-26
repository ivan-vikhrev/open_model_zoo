# Tests for demos

## Test cases

Tests cases are represented as `yaml` file.

For any demo, which user want to test, he should specify the next sections:
 * `name` - name of demo
 * `parameters` - parameters of demo
 * `cases` - test cases for this demo

Let's consider each section separately:
1. `name` section. User can test any demo applications from OMZ `demos` directory. To get more information about demos please see [DEMOS](..).
2. `parameters` section. For this section user shoud type the next parameters:
   * `implementation` - possible values are `cpp` and `python`. Parameter is required.
   * `model_keys` - model keys from demo options. Optional parameter, `-m` flag for single model demos is used by default.
   * `device_keys` - device keys from demo options. Optional parameter, `-d` flag is used as default.
   * `parser_name` - parser for output of tested demos. Optional parameter, `basic` parser is used by default. All possible parsers:
     * `basic` - this parser does nothing. In this case, we check correct of demo working and don't consider what demo return.
     * `perf` - this parser examine demo output for performance metrics, such as `FPS`, etc. In this case, after running demo tests, parser create `.csv` file with table of performance.
3. `cases` section. In this section user spicifies demo options for test cases. There are some rules for creating demo test cases:
   * Options located in *one-level* indentation will be combine with each other.
   * User can define list of different values for single key-option. It means that demo test cases will be created according to this list.
   * If user want to separate difficult (many-options) cases and test them independently, he should use key-word `split` for these cases.
   * For python demos user should use *long* keys for options.

   Example:
   ```
   - name: human_pose_estimation_demo
     parameters:
       implementation: cpp
     cases:
       no_show: true
       input: !DataPattern human-pose-estimation
       split:
         - architecture_type: openpose
           model: !Model human-pose-estimation-0001
         - architecture_type: higherhrnet
           model: !Model higher-hrnet-w32-human-pose-estimation
         - architecture_type: ae
           model:
             - !Model human-pose-estimation-0005
             - !Model human-pose-estimation-0006
   ```

   Based on config above, the next `4` test cases will be created:
   ```
   TestCase(options={'-at': 'openpose', '-m': <args.ModelArg object at 0x7f6a70590580>, '-no_show': True, '-i': <args.DataPatternArg object at 0x7f6a70584580>}, extra_models=[])
   TestCase(options={'-at': 'higherhrnet', '-m': <args.ModelArg object at 0x7f6a705905e0>, '-no_show': True, '-i': <args.DataPatternArg object at 0x7f6a70584580>}, extra_models=[])
   TestCase(options={'-m': <args.ModelArg object at 0x7f6a70518910>, '-at': 'ae', '-no_show': True, '-i': <args.DataPatternArg object at 0x7f6a70584580>}, extra_models=[])
   TestCase(options={'-m': <args.ModelArg object at 0x7f6a70518970>, '-at': 'ae', '-no_show': True, '-i': <args.DataPatternArg object at 0x7f6a70584580>}, extra_models=[])
   ```

   What about custom types for `input` and `model` options, there are next alternatives:

   For `input` flag:
   * `string` type - scpecify path to image or folder of images directly.
   * `!TestData` tag - specify a relative path to input data, based on `test-data-dir` flag for `run_tests.py`.
   * `!OmzData` tag - specify a relative path to input data, based on `OMZ_DIR`.
   * `!image_net_arg` tag - specify a relative path to images from imagenet dataset. User must use image `id`, as input for this tag.
   * `!image_retrieval_arg` tag - specify a relative path to images from image retrieval dataset. User must use image `id`, as input for this tag.
   * `!brats_arg` tag - specify a relative path to images from BraTS dataset. User must use image `id`, as input for this tag.
   * `!DataPattern` tag - specify one of data sequence names from [data sequences](data_sequences.py). Use image patterns as input for demo.
   * `!DataDirectory` tag - specify one of data sequence names from [data sequences](data_sequences.py). Use folder with images, that have pattern name format, as input for demo.
   * `!DataDirectoryOrigFileNames` tag - specify one of data sequence names from [data sequences](data_sequences.py). Use folder with images, that have original names, as input for demo.
   * `!DataListOfFiles` tag - specify one of data sequence names from [data sequences](data_sequences.py). Use list of names of images, as input for demo.
   * `!Vector` tag - specify list of any types above.

   For `model` flag:
   * `!Model` tag - specify name of IR model. Also user can omit this tag and specify name directly.
   * `!ModelFile` tag - scpicify dict, where should be next (key: value) pairs:
     * `name`: name of IR model.
     * `file_name`: name of file from folder of model.

     This type is useful for models of other frameworks obtained during conversion
   * `null` type - user should specify this type if he doesn't want to use some model in current case.

   For other flags: default yaml types + all types above if needed.

   To get more information about types, please see [types](args.py).
   There is [`default_config`](default_config.yml) to test all demos. Please look to it, if you want to create custom yaml config.
## Run tests

To start testing you should run the next script `run_tests.py`.
Running the script with the `-h` option yields the following usage message:

```
usage: run_tests.py [-h] --demo-build-dir DIR --test-data-dir DIR --downloader-cache-dir DIR [--config CONFIG] [--demos DEMO[,DEMO...]] [--scope {base,performance,custom}] [--mo MO.PY] [--devices DEVICES]
                    [--report-file REPORT_FILE] [--log-file LOG_FILE] [--supported-devices SUPPORTED_DEVICES [SUPPORTED_DEVICES ...]] [--precisions PRECISIONS [PRECISIONS ...]] [--models-dir DIR]
let's consider each section separately
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
  --config CONFIG       The config file with test cases
  --demos DEMO[,DEMO...]
                        list of demos to run tests for (by default, every demo is tested). For testing demos of specific implementation pass one (or more) of the next values: cpp, cpp_gapi, python.
  --scope {base,performance,custom}
                        The scenario for testing demos.
  --mo MO.PY            Model Optimizer entry point script
  --devices DEVICES     list of devices to test
  --report-file REPORT_FILE
                        path to report file
  --log-file LOG_FILE   path to log file
  --supported-devices SUPPORTED_DEVICES [SUPPORTED_DEVICES ...]
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
