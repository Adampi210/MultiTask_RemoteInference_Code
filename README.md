# MultiTask_RemoteInference_Code

## To run the code
1. Run the command <code>sh dset_prep.sh</code> to download the NGSIM dataset (by default downloads the lite version with few files, to download the full one modify the dset_prep.sh file to have NGSIM_data_URL_list.txt instead of NGSIM_data_URL_list_lite.txt).
2. Then create the new conda environment using the command <code>conda env create -f drone_video_analysis_env.yml</code>.
3. Activate the environment using th command <code>conda activate drone_video_analysis_env</code>
4. Test the scheduler and frame extractor using the command <code>python3 data_funcs.py</code>
5. Test the segmenter using the command <code>python3 NN_funcs.py</code>