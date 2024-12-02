# AI Palooza Competition Guide 2024: Deep Racer 2.0
Welcome to the AI Palooza Competition 2024! Please thoroughly read this document before posting any questions on the forums!

## Setup
Let's set up your analysis environment!

1. Install VSCode by visiting the Software Center or Leidos App Store on your computer. If you already have an IDE you
   prefer that's fine, however, you will need to access Jupyter notebooks so just make sure your IDE supports this.
2. If you do not have Git installed on your system, visit the Software Center or Leidos App Store to install it on your
   computer.
   - Git is a distributed version control system designed to handle everything from small to very large projects with speed
  and efficiency. It allows multiple developers to work on a codebase simultaneously by tracking changes, coordinating
  work, and managing project versions. Git provides robust features for branching, merging, and tracking history, making
  it an essential tool for collaboration in software development. It ensures code integrity and facilitates version
  control, making it easier to manage project progress and collaborate on code changes.
  - On Mac / Linux git will work from the terminal once installed. On Windows you will need to use the "Git Bash" (found
  via searching with the Windows Key) or you may use the Miniconda 3 Powershell.
3. Clone this repository in a location of your choosing.
4. Open VSCode to the deepracer-contestant folder, navigate to README.md, access the command palette by using
   `CTRL+Shift+P` on Windows/Linux or `CMD+Shift+P` on Mac, type in "markdown" and then select open preview to the side.
   Now you may view this document in a more readable format.
5. Install Miniconda 3 using the quick install command line instructions found
   [here](https://docs.anaconda.com/miniconda/#quick-command-line-install).
- Miniconda 3 is a minimal installer for Conda, a package manager and environment management system. It includes only
  Conda, Python, and their dependencies, providing a lightweight alternative to the full Anaconda distribution.
  Miniconda 3 allows users to create isolated environments and install additional packages as needed, making it a
  flexible and efficient tool for managing Python projects and dependencies.
- On Windows you may install using the command terminal or powershell. Hit the Windows Key and type in "cmd" or
  "powershell" then follow the instructions on the webpage.
- On Mac / Linux you may use the terminal. On Mac hit CMD+Space and type in "terminal".
6. Create the Conda Environment.
- On Windows hit the Windows Key and search for "miniconda", you should see an option to access the Miniconda
  PowerShell. Use this to interact with miniconda and git moving forward. 
- On Mac and Linux, the normal terminal can access miniconda.
- In your respective terminals, navigate to this folder by using `cd <path_to_here>`. You may get the path by right
  clicking in the file tree on the left side of VSCode and clicking copy path".
- Once at this folder, run the command: `conda env create -f environment.yaml`
- If you need to run any python files from the terminal or miniconda 3 powershell, make sure the conda environment is
  active first by typing in the command `conda activate deepracer-env`. You should see a `(deepracer-env)` prefix now.
  To deactivate the conda environment within your terminal, use `conda deactivate`, you will see the `(deepracer-env)`
  prefix disappear.
6. In VSCode use `CTRL+Shift+P` on Windows/Linux, or `CMD+Shift+P` on Mac to access the VSCode command palette, type
   "Interpreter" and select `Python: Select Interpreter`, choose `deepracer-env`. You should only have to perform this
   once.
7. Open `install_requirements.ipynb`. In the upper right corner click `Select Kernel`, select `Python Environments`,
   then select `deepracer-env`. This will ensure that your Jupyter Notebook runs using the Conda environment we just
   created.
8. Run the python cells inside `install_requirements.ipynb` to install all additional libraries we need for this
   competition.
9. If you want to run any of the `.py` files, you may do so from the terminal or miniconda 3 powershell by first making
   sure the `deepracer-env` environment is active.
   - On Windows type in `Get-Command python`, if the `Source` on the right-hand side is something like
     `...\miniconda3\envs\deepracer-env\...` that is confirmation your python command is pointing at the right spot. If
     you see another path, try `Get-Command python3`. 
   - On Mac/Linux type in `which python` or `which python3`, you want the one that results in a path looking like
     `.../miniconda3/envs/deepracer-env\...`.
  - To run a python script (`.py` file), use the command you have verified like this: `python <filename>.py`.
  - You may also try to navigate to the Python file in VSCode. In the bottom right corner click the python version, then
    select `deepracer-env`. You can alternatively use the command palette to change the Python interpreter. Then right
    click inside the file and select `Run Python>Run Python File in Terminal`. Or you may simply use the VSCode terminal
    and the above commands (open the VSCode terminal with CTRL+`, on any system). Windows users will need to potentially
    edit the VSCode settings to use the miniconda powershell if they want to do everything from VSCode.

## Competition Overview
Welcome to an exciting exploration into the world of Reinforcement Learning (RL)! In this project, you'll have the
chance to experiment with an RL algorithm using AWS Deep Racer and gain hands-on experience by observing and studying
its performance under different circumstances. Additionally, if you are on an Intel machine you will get to try out some
exciting optimized libraries that will greatly enhance your productivity! This experiential learning is at the heart of
Reinforcement Learning, and it will help you grok your algorithm deeply.

While this competition is a simulated project, we encourage you to approach it with the seriousness and dedication of a
real-world scenario. Your work here at Leidos has the potential to impact real people. We will support you in systematic
experimentation as well as ethical analyses of your procedures and results. We want you to consider the real-world
implications of your work based on the results you achieve and the questions we pose. Your journey will be guided by a
strong focus on scientific and ethical justification, which is crucial for success as an AI/ML practitioner.

Through this project, you will be introduced to:
1. AI | Machine Learning | Reinforcement Learning
2. Scientific Thinking
3. Ethical Thinking

Our goal is for you to truly grasp the practice of Reinforcement Learning, which involves the interaction of algorithms
with environments. While you will be optimizing an RL model in Deep Racer, much of what you will learn comes from
conducting your own analysis of data in the form of a report. Over the next eight weeks you will experiment with and
analyze an RL model across potentially two different virtual racing tracks.  Using the template we've provided, you will
construct a detailed report of your experiences and findings. As for your optimized model, we will evaluate that on its
ability to drive the vehicle around the track in a timely manner, as well as its ability to keep the vehicle stable. 

At the end of the first 8 weeks in the competition (the open phase), you will submit your best model along with your
report. We will review your submissions and select finalists to advance to a closed phase of the competition.  A four
week closed phase will follow the open. Finalists will optimize their models and reports further, culminating in a
thrilling live event at our Global Headquarters in Reston, VA. Here, the finalists' models will be deployed on real
vehicles to race around a physical track. Additionally, finalists will have the opportunity to present their work during
the live event, allowing for practice and showcase of technical communication skills. 

Throughout both phases, we have integrated ethical exercises for you to complete. These will help you analyze your
procedures and results, considering their ethical implications. Your insights and engagement in these exercises are
highly anticipated. Additionally, you will have the opportunity to experience the new Leidos Governance tool, OneTrust.
All AI projects at Leidos will be required to utilize this tool in the examination of the project. You will provide
valuable feedback on the tool and questions and shape the future of Leidos AI Governance. We look forward to your deep
insight and critical assistance!

For more information on grading, see [the grading section](#grading).

**You will have a maximum of 48 hours of compute time during the open phase** A suggested pace for the open phase is 6
hours per week of compute time. A good division of your hours would be 18 hours maximum for exploration, leaving 30 hours for
optimization. Model performance has decent behavior by 3 hours, competitive behavior at 6 hours, and the onset of
diminishing returns at 10 hours. 18 hours allows 3 hours per experiment for exploration which is enough to
observe meaningfully different behavior. You will then have up to 3, 10 hour runs for step 3 optimizaion. 
- Note: When we ran our own analysis to determine suggested times for you, we observed reasonable differentiation in
  metrics at the 3 hour mark. However, on some runs we ran for 6-10 hours, meaningful differentiation also occurred that
  was not present at the 3 hour mark. While we are limiting you to 3 hours per run for step 2 (please to not go beyond
  this), we wanted to inform you of our observations to 1) let you know how we got these numbers and 2) to provide with
  you with a helpful anecdote. When you are on your own projects you will be the one to determine "how much time is good
  enough", and sometimes you will see behavior early, and other times it will take a while. 

Because there is limited compute, we will have a registration cap of 50 with a waitlist of 20.
Registered contestants after 3 weeks of inactivity will be removed from the competition and waitlisted individuals will
take the dropped spot.  To be competitive, you will want to spend a few hours on this each week.

Please post any questions you may have on the Forums. Due to the scale of this competition, and feedback we have
collected in previous years, this year we are we are requiring the use of our new forums for communication. The forums
allow us to better facilitate answering your questions and they allow others to benefit from the questions to ask to us,
and one another!

Lastly, we will hold office hours every Tuesday and Thursday of the competition. Series 1 is on Tuesday and Series 2 is
on Thursday, they run at different times. Please check the [AI Palooza Competition
Calendar](https://prism.leidos.com/technology/enterprise_events/ai_palooza/ai_palooza_challenge/ai_palooza_challenge_calendar)
to see which slot works best for you.

We truly hope you find this experience enriching and enjoyable, and that you come to "grok" the fascinating field of
Reinforcement Learning. Happy learning and let us grok together!

```
Grok:

"Grok" is a term that means to understand something deeply and intuitively. It was coined by Robert A. Heinlein in his
1961 science fiction novel *Stranger in a Strange Land*. In the book, it is a Martian word that means to drink, but it
is also used to signify a profound level of understanding, akin to merging with or becoming one with the subject being
understood. When you grok something, you grasp it completely in a way that goes beyond just intellectual comprehension.
```

## Competition Agenda
### Registration
- June 17-28
### Open Phase
- July 1: Competition Launches
- August 9-12: First Virtual Competition
- Mid-August: Governance Questionnaire Released
- August 30-September 2: Open Phase Final Submission Window
- September 6: Finalists Announced
### Closed Phase
- September 6: Closed Phase Launched
- September 20-23: Closed Phase Final Submission Window
### Competition Finale
- October 3

## Prizes

We have exciting prizes for finalists based on four categories: model performance, scientific analysis, ethical
analysis, and collaboration. There is no singular first place, as multiple paths to prizes are provided. There are
prizes for finalists and non-finalists. Additionally, we will potentially provide swag to contestants. However, all
prizes (i.e., items we grade your work for) are quite nice and substantial, the swag is just an added thing on top 😄!

Prizes:
- High-Performance Intel Laptops
- High-Performance Nvidia GPU's

Swag:
- TBD

### Raffle Series
We are hosting a participation-based raffle series. There will be three raffles, each with one opportunity for entry:
- Submitting to the first virtual race on August 9-12
- Completing our Governance Questionnaire by August 26. Questionnaire will be sent out mid-August (exact date TBD)
- Submitting to the final window on August 30 - September 2

Each action you complete is an entry into the raffle associated with that event. At the competition Finale we will
select three winners via random draw for each raffle making a total of nine prize winners. Raffle winners may only
receive one prize each. Competition finalists may not participate in the raffles as they have access to the finalist
prize pool. We are in the process of finalizing raffle prizes, but rest assured they will be very nice.


## Competition Instructions and Guide
The heart of this project is your journey of exploration, tuning, and analyzing. This is not just a task, but an
opportunity to deeply engage and learn. The thorough and insightful analysis that you provide in your report will be the
cornerstone of the competition. For all works that you draw upon for your analysis, you must cite your sources. Please
see the section on [plagiarism and citations](#plagiarism-and-citations) for more information.

**It is often most helpful to conduct all work with your scientific report in mind. In fact, we strongly encourage you
to write your report while you carry out the experiment tasks competition!** For helpful tips on scientific
reading/writing, please check out [recipes for success](#recipes-for-success) 5 and 6.

Enjoy the process and let your curiosity and dedication shine through!

### Open Phase

In your first 8 weeks you will:
1. Select and grok an RL algorithm
2. Optimize the RL model on Deep Racer
3. Transcribe your journey.

Complete all tasks below and fill out the `report.ipynb` thoughtfully.  You may also choose to use the template to write
a report in Word or LaTeX if you would like.  You will need to submit a report that demonstrates engagement and grokking
to progress to the finale. 
- As an aside, LaTeX is pretty great for technical writing so even though you may not use it here, look into it for
future use! It's basically coding but for tech writing, allowing you to focus on content instead of formatting.

**You will use the 2022 reinvent champ track until the first virtual race on August 9. Make sure to select this track in
the Deep Racer console! Do not select any other track. You will be given a new track after this!**

There is also a *prize* for the best report, regardless of whether your Deep Racer car makes it into the finale or not.

Please reference [good and bad analysis](#example-of-good-and-bad-analysis) so you know what type of analysis and
insight we are looking for! There is additional information further in this README.md that will also help you in your
journey of creating a solid report 😄.

#### Step 1. Getting familiar with the Deep Racer Repository and this Repository.
We know you are engaging in this in your free time, so we hope to provide some easy ways to engage with lots of room to
dive deep and learn a lot. Head to the Deep Racer console on AWS and poke around.  Build some momentum by training a
model on all default values for ~10 minutes. This should take a few minutes to set up. After your initial model
is finished training, run an evaluation session. Congratulations! You now have the familiarity and necessary files use
the code! Download your training log, your physical model, and your evaluation log! These will all come in the form of a
tar.gz.

Your next task is to read through `deepracer_analysis.ipynb` and `make_graph.ipynb`. Here you will observe how to work
with these files!

Your next task is to read through all of `report.ipynb`, so you get a feel for the questions we want you to answer.

Next, take a look at `build_submission.py` and `evaluation.py`, so you can check out how to turn in your model and
what we are collecting to grade your model! We will only officially grade your model at the end of the open
phase, but throughout the summer we will have virtual races where you can submit your best model so far and see how
it stacks up against others!

Lastly, answer Starter Questions, and Question 1 on `report.ipynb`.

*All the code you need to train, evaluate, and submit and initial model is written for you.  If you aren't familiar with
python, you can use this as an opportunity to read through and understand it better, or you can focus on understanding
the algorithm and just run the Jupyter notebook.  If you are familiar with python, we welcome you to mess around with any
of the code to see how you can improve things or add your own personal flavor.*

#### Step 2. Initial Exploration & Experimentation
The relevant report section is `Initial Algorithm Exploration & Experimentation`

**We ask that in this
step, you do not change the reward function, or any other settings outside of the hyperparameters you are adjusting.**

**You will complete step 2 using the 2022 reinvent championship track! Make sure to select this track in
the Deep Racer console! Do not use any other track!**

##### Concrete Procedure for Step 2
1. Grok both algorithms
2. Select one algorithm to move forward with and construct a hypothesis about that algorithm for the Deep Racer
   Environment.
    - Hypothesis: A hypothesis is an assumption, an idea that is proposed for the sake of argument so that it can be
      tested to see if it might be true. In the scientific method, the hypothesis is constructed before any applicable
      research has been done, apart from a basic background review.
    - Base your hypothesis on what you know about the algorithm. Write something that you can confirm or deny based on
      empirical experimentation within the Deep Racer environment.
    - Your hypothesis will allow your analysis to become individualized. As such, there is no perfect hypothesis, so
      don't spend too much time here, the important thing is to get something down and begin analyzing through that
      lense. Also, know that its okay to revise or change your hypothesis wholesale later on if you feel the need to
      based on something you learn or observe 😄.
3. Train your model with different hyperparemeter values, 2-3 hours per training session maximum. (see [explanatory
   notes](#explanatory-notes-for-step-2)).
4. Create training graphs for hyperparameter values across the two metrics: reward, distance to center.
5. Analyze your graphs. Interpret and explain them. Connect your insights to the problem, to the hypothesis, to the
   algorithm, etc.

##### Explanatory Notes for Step 2

In the first few weeks you will 1) construct a hypothesis based an algorithm and the Deep Racer environment, 2)
experiment with the algorithm upon that environment, and 3) analyze and explain your experiments. **We ask that in this
step, you do not change the reward function, or any other settings outside of the hyperparameters you are adjusting.**
This will allow you to observe the particular effects of the individual hyperparameter upon the algorithm. Your goal
here is to learn about the algorithm you selected empirically by watching how its hyperparameters affect it within the
Deep Racer environment.

Learn about these two algorithms. You will need to pick one and then move forward. The algorithm you pick, you will use
for the entire competition. See these [Helpful Algorithm Resources](#helpful-algorithm-resources) for introductory and
advances content on the Algorithms!
- Proximal Policy Optimization
- Soft Actor Critic 

Once you have selected your algorithm, answer Question 2 in `report.ipynb`.

**Only experiment with one algorithm.**

Next, construct a hypothesis about that algorithm. Go ahead and answer Question 3 in `report.ipynb`. Make sure to back
up your hypotheses with experimentation and thorough discussion later on.

You will tune the two hyperparameters for the algorithm you select:  
- PPO Hyperparameters: 
  - Entropy
  - one more hyperparameter of your choosing.
- SAC Hyperparameters:
  - SAC alpha (α) value
  - one more hyperparameter of your choosing.

- For the second hyperparameter you may select from the following:
  - Learning rate
  - Discount factor

You will use the code in make_graph.ipynb to plot the performance of your algorithm across your hyperparameters according
to reward and distance to center.  You will create "training graphs" from your training logs that demonstrate how your model
learns over time. Training graphs have the number of episodes on the X-Axis and metric values for the Y-Axis; they can contain
hundreds of episodes on the X-axis. See `make_graph.ipynb` for example graphs already displayed. 
We suggest training at least 3 values per hyperparemeter.  This means you will have at least 6 models all using the same reward
functions and other settings, outside of the one hyperparameter value change. You can compactly use one graph to represent your
entire hyperparameter sweep for one metric and hyperparameter combination. Therefore, you will end up creating at least four graphs
from your 6 trained models.

A decent model could be achieved within 3 hours of training. A well-performing model can be achieved
after 6 hours, and after 10 hours you likely won't see any more improvement. We suggest training for no more than 3
hours for each experiment during this step, so that you do not burn through your compute credits! In other words, 3
hours per hyperparameter value, for a maximum of 18 hours total computation time.

`make_graph.ipynb` is where you will create your graphs btw, and the code is already written for you! All you need to do
is run your training sessions, download the logs (they will be `tar.gz`'s), put them into `make_graph/`, name them
accordingly (check out how we named the ones we provided for you in `make_graph/`), and plot away!

Once your graphs are created, you will analyze, explain, and interpret these graphs based on 1) what you have learned
about the model, 2) what you have learned about the problem, and 3) what you stated in your hypothesis.

See [good and bad analysis](#example-of-good-and-bad-analysis) to view the type of analysis we would like you to perform
based on your graphs and experimentation!

When you design your hyperparameter range, base your range upon your research into the algorithm you have conducted
through the resources we have provided or based upon resources you have found. You should seek to explain why you chose
the values you did, intuitively.

Also, throughout this step, make sure to progressively fill out the `Initial Algorithm Exploration` section in
`report.ipynb`. Its best to iterate over this section as you work, rather than leaving all to one sitting!

#### Step 3. Optimization!
The relevant report section is `Optimization`. The algorithm / model you ultimately create here is the one you will
submit for candidacy into the final phase.

**Prior to August 9 you will use the 2022 reinvent championship track! Make sure to select this track in the Deep Racer
console! Do not use any other track! After August 9 you will receive a new track. Your final model must be produced on
the post-August 9 track!**

##### Concrete Procedure for Step 3
1. Experiment with as many hyperparameters and hyperparameter values as you like, as many reward functions as you like,
     and whatever action spaces you like until you are satisfied with performance.
2. Graph your results, for reward and distance to center, in training graphs and evaluation graphs (the evaluation graph
   must be over 5 trials). You should have no more than 4 graphs, 2 training and 2 evaluation.
3. Analyze, explain, and justify your reward function design and choices, (it's okay if its the default, but a good
   explanation and justification of the function is still required!). Connect your insights to the problem, to the
   hypothesis, to the algorithm, to step 2. Be sure to describe why you are satisfied, and justify why you are
   satisfied. 
4. Analyze, explain, and justify your action space design and choices. (it's okay if its the default, but a good
   explanation and justification of the action space is still required!). Connect your insights to the problem, to the
   hypothesis, to the algorithm, to step 2. Be sure to describe why you are satisfied, and justify why you are
   satisfied. 
5. Analyze, explain, and justify your results in your graphs. Connect your insights to the problem, to the hypothesis,
   to the algorithm, to step 2, anything you can think of. Be sure to describe why you are satisfied, and justify why
   you are satisfied. 
6. Create a Grad-CAM image from your trained model.
7. Explain the Grad-CAM as a function of your decisions, the algorithm, the reward function, the environment, anything
   you can think of! Explain everything you can in the image and justify everything you say.

##### Explanatory Notes for Step 3
Your goal here is to dive deep into the analysis of your optimization. Be sure to ask as many questions as you can,
provide as many answers as you can, and extract all the insight you can out of your results!

Now that you have seen and explained the effects of two hyperparameters on the your algorithms, you are free to select
any hyperparameter values that you like! Additionally, you may change your reward function as you see fit! And you can
change your action space to whatever you please! 

When you are happy with your new performance, provide training graph(s) and evaluation graph(s). Once your graphs have
been created, provide subsequent meaningful analysis explaining and justifying your decisions and results. Also compare
and contrast with the results you achieved in step 2. You must have at least one training graph and one evaluation graph
in this section. Your experience here will be largely the same as step 2, but with more freedom in your hyperparameter
choices, the new goal of optimizing for performance instead of only seeking to 'get a feel' for your algorithm, and the
requirement of an evaluation graph.

An evaluation graph differs from a training graph in purpose. An evaluation graph demonstrates your model's
performance in a session where no learning takes place, you are just seeing how it does. It's meant to be a
representation of how your model would act in your environment if you deployed it with a particular training state
achieved. In the AWS Deep Racer console, evaluation is easy! Just tap evaluate model on the model you wish to
evaluate (you have probably already observed the AWS prompts to do this), and make sure to run the evaluation over 5
trials. Once your evaluation is done, you may watch the video recordings of how your model did! Plotting an evaluation
is easy, just download the logs as before, put them into `make_graph/`, and run the plotting code!

You will also need to explain your reward function and your action space. Be sure to record all decision points that
lead you to your ultimate space and function, so its easy for you to fill out your report and justify everything you
decided to do!

You will be creating a Grad-CAM image in this step. Grad-CAM stands for Gradient [-weighted] Class Activation Mapping.
In short, it checks out what your spatial model focused on in the input image by taking special note of the gradients.
Grad-CAM images are heatmaps that let you see where your model is focusing! You will explain your unique Grad-CAM image
as a function of your action space, your reward function, your hyperparameter choices, and anything else you can think
of that might be relevant!
- [Check out the original paper here!](https://arxiv.org/abs/1610.02391) The original paper has an awesome introduction
to AI explainability.

As in Step 2, please make sure you are iteratively filling out the relevant report section, `Optimization`. It is much
better to work iteratively in technical writing than in one big single session!

#### Step 4. Ethical Considerations & Future Improvements
Relevant section in the report: `Ethical Considerations`. You must complete all prior sections in order to work this.
Your answers depend upon your completed results from Step 3.

Please work through all ethical sections in the report template. Connect everything you think relevant from step 3 to
your answers here, e.g., grad cam, reward function, results, analysis, etc. Please provide thorough discussion and think
deeply here.

#### Step 5. Contemplative Reflection.
Relevant section in the report: `Contemplative Reflection`. You must complete all prior sections in order to work this.
Your answers depend upon your completed results from Step 2 and 3.

Please work through all contemplative reflection. Think over everything you have done thus far and provide thorough
discussion.

#### Step 6. AI Governance Tool
For all AI projects at Leidos, employees will be required to work with the new AI Governance Council. Part of this
includes the usage of a new AI Governance Tool, OneTrust. We will send you all OneTrust links where you can access a
special Questionairre. This is going to be the exact same tool and questionairre you will utilize if you contribute to
an AI project at Leidos. 

You will have the opportunity to familiarize yourself with this tool and utilize it as if this competition were a real
Leidos project. Additionally, you will have the opportunity to provide critical feedback on the tool and the set of
questions. Your feedback is extremely important and valuable. You will help shape AI governance at Leidos, which will
affect thousands of employees, hundreds of projects, and untold numbers of end users.

Be on the lookout for our access links. We will provide this to you late in the open phase.

#### Step 7. Open Phase Submission
At the end of the open phase you will submit:

- your final model
- associated 5-trail evaluation logs (use build_submission.py)
- final report
- your completed governance OneTrust form.

We will grade your work select our (8) finalists who will move onto the closed phase. Additionally, we will determine
who will win prizes among the non-finalists.

##### Submission Requirements and Procedure.
To submit your work please use `build_submission.py` and follow the instructions within. 

### Closed Phase
Closed phase instructions will be provided during the transition week from open to closed. We can't reveal all our
secrets just yet 😉.

## Grading
The overwhelming emphasis of our grading scheme is on your grokking and learning journey, i.e., the report.

To this end and we will ensure a fair evaluation of your model performance. You do not need to have the overall best
performing model to move into the final phase or to win prizes. That is to say, we aren't dismissing model
performance, it will be a part of the grading process, but if you did not happen to select the magic ultimate
combination of hyperparameters, but still provided solid analysis and insight in your report it will all balance out!

Additionally, re: your insight and analysis. We will grade you based on whether the internal logic of your insights
makes sense and that you attempted to provide meaningful insight based upon what you have learned about the models
empirically through Deep Racer usage, and through study of our provided resources / the resources you have looked up on
your own. This is a learning exercise and we want you to have fun without pressure. There is no totally wrong answer,
only a totally wrong type of answer!

Lastly, your final model must be produced on the post August 9 track!

## Virtual Competition Instructions
We will host two virtual competitions to update the leader board.
Participants can submit whatever model they have through an AWS portal at any point over a 72 hour span starting on Friday
and ending Monday morning. Each race will be on a new track to encourage the development of robust models.

Open Phase:
- You will focus on the first track for the first four weeks and the second track for the second four weeks. The second
  track is what you will be graded on for entrance into the final phase.
- You will work through two tracks, the 2022 reinvent championship track up until August 9, and a new track after this!
- When its time to submit your model you will upload it to the virtual race submission portal. You will also utilize
  `build_submission.py` to process your 5-trail evaluation logs for us. To check that your submission was built
  correctly you can use `evaluation.py`, this is the file we will be using as well.
- Read through `build_submission.py`

### Example of Good and Bad Analysis

The following graph was generated over a stratified 5-fold cross validation for each dataset size. A Support Vector
Machine was progressively optimized with larger and larger slices of an unknown dataset. That is to say, the SVM was
trained on a small portion of a dataset then evaluated. The portion of the dataset was then increased, and then the
model was trained and evaluated again - using 5-fold stratified cross-validation the entire time. The point of this
procedure is to examine how the algorithm performs under varying amounts of data from the training set. Determinations
can be made about how sample efficient, or inefficient an algorithm is upon the problem with exercises like this.

This graph is known as a "learning curve". Let's examine two analyses of this learning curve. 

![learning-curve](readme_images/learning_curve.png "Learning Curve")

#### Lackluster Analysis
The algorithm, despite receiving more samples progressively, does not gain much performance. At a sample size of ~50
instances the recall is quite high, .95 and .97 for both training data and validation data respectively. As the sample
size increases the recall score drops over 10 points, then remains roughly the same throughout all dataset sizes.

The precision score at a sample size of ~50 instances is just higher than random performance, .45 and .47 for training
and validation data respectively. The precision score then raises to ~.55 as sample size increases to 100-200 instances,
then raises again slightly to flatline for the remainder of the dataset sizes. 

For both recall and precision a sample size of 200 instances appear to allow the model to reach its relative peak
performance on each metric, without any significant change beyond those sample sizes.

There is a high degree of variance in the recall scores of the training sets, while in contrast the precision training
set scores have a tight standard deviation throughout the entirety of the optimization procedure.

Lastly, there is slight a slight variance in the reported curves where the model is demonstrated to be slightly
underfit.

#### Better Analysis
The SVM algorithm operates by finding the optimal decision boundary that maximally separates different classes. As
demonstrated by Fig. 1 the optimal SVM decision boundary for this dataset is found within ~200 stratified training
instances. That more training instances do not provide information which would change the optimal decision boundary
suggests that the feature space, or underlying data structure is sufficiently demonstrated with low stratified instance
counts relative to the total number of available samples. Further analysis of a t-SNE plot generated from the dataset
would confirm or reject this hypothesis.

However, despite the low threshold for an optimal decision boundary, recall has high variance throughout the procedure
demonstrating the complexity of the prediction task and the insufficiency of the data to represent the prediction task.
The information / feature space contained within the dataset is sufficiently expressed with low instance counts, but the
data does not contain sufficient information to allow the SVM to remember positive instances optimally. Precision has a
low variance throughout the training procedure demonstrating that there is no subset of positive instances which are
consistently remembered.

The decision boundary that is formed by the SVM is the best possible with the feature space available, demonstrated by
the consistent performance after ~200 instances, but the lack of additional diverse data prevents the decision boundary
from being formed such that positive instances are consistently and correctly identified, demonstrated by the low
precision score and the high recall variance.

Thus, additional, diverse data is required to represent the decision space to sufficiently solve the problem.

#### Analysis Takeaways
Essentially, the difference between the two analyses is one answered "why" and provided grounded interpretation,
attempting to connect the whole of the situation at hand to the interpretation, while the other merely described the
image in text form. Better analysis interprets the data in a holistic fashion, better analysis does not restate the
visualized.  Your graphs/plots can speak for themselves, we want you to provide insight based on what the plots are
saying.

Additionally, the better analysis revealed something about the algorithm and about the problem. You should seek to
replicate this in your own analysis.

## More Details

### Training Time
The initial model takes about 3 hours for training to achieve decent results. After 6 hours you will achieve
competitive behavior. After 10 hours you will achieve diminishing returns.

### What makes an excellent report?

An excellent report has an thoughtful, in-depth analysis that demonstrates experimentation and analysis that guided the
results.  Think quality over quantity.  You do not have to write multiple paragraphs for each answer.  Instead, consider
what meaningful questions you can answer.

### Questions to consider for analysis

- Why did you get the results you did? Compare and contrast the different algorithms.
- What sort of changes might you make to each of those algorithms to improve performance?
- How long did each combination take to become optimal and why?
- What did not result in optimality and why?
- How did the hyperparameter values affect the performance and why?
- How much performance was due to the hyperparameters you chose?
- Which algorithm performed best?
- How do you define best?
- How do you know your algorithm is the best that it can be, or should be, and why?
- Which algorithm did you ultimately select for your vehicle and why?
- What did or did not line up with your hypotheses and why?
- How much performance was due to your reward function design?
- What behavior did your reward function cause?
- Did your reward function cause any unintended behavior?
- Does reward affect each algorithm the same, why or why not, and what does that mean for the behavior you observed?
- How will you address errant behavior?

Be creative and think of as many questions you can, and as many answers as you can.

### Extra Credit Opportunities in the Open Phase
- You will suggest improvements to your algorithm through the lense of what you have empirically discovered and
  determined. You do not have to create or invent these improvements.  Both PPO and SAC have been analyzed and discussed
  thoroughly within the scientific community. Your job is to use your findings to guide your research online, find
  papers with your improvements discussed, and provide your commentary on the improvements. Specifically, provide your
  own interpretation of the improvements by discussing the methods, the experiments, and the results of the authors. We
  will assign you meaningful extra credit for completing this optional section.

- Read 1-3 papers or blog posts on different reinforcement learning techniques and provide a 5 sentences minimum, 10
sentences maximum on how the technique works, and how it might be used to improve your model's performance.  You can use
Arxiv, Google Scholar, or Semantic Scholar (❤️) to search for resources, as well as repositories like this one 
https://github.com/yingchengyang/Reinforcement-Learning-Papers

- Complete the bonus section in the report for the extra credit.

### Recipes for Success
1. Start early, and work at a steady pace. The key is to find a pace that you feel you could work indefinitely without
   growing tired or feeling stressed. This is a marathon, not a sprint. Small bites of work each day, every few days, or
   at least each week will let you take on this challenge and do very well!! We suggest completion of step 1 within 2-3
   weeks, leaving the next 5-6 for step 2.
2. Make use of the forums!
3. Work with the report, work through the report, work alongside the report. Keep the report in mind the entire time.
   Formulate plans and action items for the week based on what you need to do stemming from the report. Write sections
   of the report progressively, and don't be afraid to delete sections and restart sections if you need to!  
4. Keep a log! Keep a log for the entire summer. Each time you stop work, spend 10 minutes writing down your concluding
   thoughts and where you want to pick up next time. When you return to the project, you will thank yourself for your
   notes! As you start again, read through your past notes, where things left off, and recall what was on your mind.
   Have any new insights clicked while you were away? Spend 10 minutes writing down a plan of action for your new
   project session. This "startup" and "shutdown" procedure each time you work on this project will be valuable. You can
   also have a weekly startup and shutdown where you look at things more globally! What worked, what didn't work, whats
   the plan for next week? How are those plans different from last week?
5. Read: [Ten simple rules for reading a scientific paper](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7392212/)
6. Read: [Ten simple rules for structuring papers](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5619685/)
7. A good exercise is to ask yourself "why" five times, to really get to the root cause. 

### Helpful Algorithm Resources
- [PPO Scientific Paper](https://arxiv.org/abs/1707.06347)
- [SAC Scientific Paper](https://arxiv.org/abs/1801.01290)
- [Intro to Actor Critic Algorithms](https://www.youtube.com/watch?v=w_3mmm0P0j8)
- [Intro to Policy Gradient Methods](https://www.youtube.com/watch?v=5P7I-xPq8u8)
- [5 Hour Breakdown of Advanced Actor Critic Methods [SAC & PPO Included]](https://www.youtube.com/watch?v=K2qjAixgLqk)
  - Phil Tabor also has an additional video elsewhere showing SAC in PyTorch
- [Google DeepMind Intro to RL Lectures](https://www.youtube.com/watch?v=TCCjZe0y4Qc)

### Plagiarism and Citations
We will unfortunately have to disqualify you if we discover plagiarism. Using the analysis, code or graphs of others in
this class is considered plagiarism. If you copy text from other contestants, websites, or any other source without
proper attribution, that is also plagiarism. The project is designed to allow you to immerse yourself in the empirical
and engineering side of AI/RL. It is important that you grok why your algorithms work and how they are affected by your
choices. We want your unique perspectives meaning:
1. The text of the written report must be your own.
2. Your exploration which leads to your analysis must be your own.

If you are referring to information you got from a third-party source or paraphrasing another author, you need to cite
them right where you use the information and provide a reference at the end of the document. We require you to use [IEEE
conference format](https://journals.ieeeauthorcenter.ieee.org/wp-content/uploads/sites/7/IEEE_Reference_Guide.pdf).

References appear in square brackets, inside the punctuation. Let's use the PPO paper as an example.

Example Sentences:

```
[1] demonstrated <some contextualized point from the paper relevant to your analysis>.

PPO has some of the benefits of trust region policy optimization (TRPO) [1].
```

Citation in Reference Section:

`[1] J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov, "Proximal Policy Optimization Algorithms", 2017`
