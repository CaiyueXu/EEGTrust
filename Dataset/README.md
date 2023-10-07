# Dataset summary
The EEGTrust dataset consists of the participant ratings, EEG recordings, face videos, questionnaires, and trajectories of an experiment where 16 volunteers joined a human-robot collaboration game. EEG were recorded and each participant also rated the trust level each trial. For 12 participants, the frontal face video was also recorded.
# File listing
The following files are available (each explained in more detail below):
<table border="1" width="500px" cellspacing="10">
<tr>
  <th align="center">File name</th>
  <th align="center">Format</th>
  <th align="center">Contents</th>
</tr>
<tr>
  <td>Face video</td>
  <td>.mp4</td>
  <td>The frontal face video recordings from the experiment for participants 2-7,10-13,15,16.</td>
</tr>
<tr>
  <td>Original EEG</td>
  <td>.gdf</td>
  <td>The original unprocessed EEG data recordings from the experiment in .gdf format.</td>
</tr>
<tr>
  <td>Participant ratings</td>
  <td>.csv</td>
  <td>All ratings participants gave to the robot during the experiment.</td>
</tr>
<tr>
  <td>Questionnaire</td>
  <td>.csv</td>
  <td>The answers participants gave to the questionnaire before the experiment.</td>
</tr>
<tr>
  <td>Trajectory</td>
  <td>.json</td>
  <td>The game trajectory from the experiment for the participants 2,3,4-16.</td>
</tr>
</table>


# File details
## Participant ratings
The file contains all the participant ratings collected during the experiment. The file is available in Comma-separated values (participant_rating.csv) formats.
Ability, Teamwork, and Trustworthy were rated directly on a 5-point Likert scale after each trial.
## Questionnaires
This file contains the participants’ response to the questionnaire filled in before the experiments. The file is available in comma-separated values formats.
Most questions in the questionnaire were multiple-choice and speak pretty much for themselves. The questionnaire also contains the answers to the questions on the consent forms.
## Face video
Face video contains the frontal face videos recorded in the experiment for the participants 2-7,10-13,15,16. They have been segmented into trials.
## Original EEG
These are the original EEG recordings. There are 16 .gdf files, each with 64 recorded channels at 252Hz. The .gdf files can be read by a variety of software toolkits, including EEGLAB for Matlab. The channels are arranged according to 10/20 systems. Each .gdf file has EEG data from all trials for one subject, with each trial starting with a marker and lasting for 60s duration.
