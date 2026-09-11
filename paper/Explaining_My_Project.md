# Explaining My Project — Plain English Notes

These are my own notes for talking about this project out loud. Everything here is written the
way I would actually say it, not the way it is written in the report.

**Contents**
- Part 0 — The whole project in one minute
- Part 1 — Words I use, explained simply
- Part 2 — The dataset and what I did with it
- Part 3 — The report, section by section
- Part 4 — The presentation, slide by slide
- Part 5 — What is left to do
- Part 6 — 25 questions my teacher might ask, and my answers

---

## Part 0 — The whole project in one minute

Diabetic retinopathy is an eye disease that people with diabetes get. It slowly damages the
retina. The bad part is that early on you feel nothing, so by the time your vision goes blurry
the damage is usually permanent.

The way doctors catch it early is by taking a photo of the back of each eye once a year. A
trained person looks at the photo and gives it a score from 0 to 4. Score 0 means healthy.
Score 2 or higher means the person needs to see a specialist — we call that "referable".

The problem is there are way more photos every year than there are trained people to look at
them. That is the gap I am trying to fill with a computer model.

So what I built is a deep learning model that looks at an eye photo and decides two things:
what grade it is, and whether the person needs to be sent to a specialist. Then I spent most of
my time figuring out what actually makes that model good, and checking that my own results were
not lying to me.

**My final model gets 0.887 AUROC on the referral decision.** In plain terms: if I set it up to
catch 90% of the people who need referral, it correctly clears about 61% of the healthy people.
That is decent but not good enough to actually use yet, and I say that clearly in the report.

---

## Part 1 — Words I use, explained simply

If my teacher asks what any of these mean, this is how I would answer.

**Grade / ICDR scale** — the 0 to 4 score for how bad the disease is. 0 is no disease, 4 is the
worst. It is the standard scale doctors use.

**Referable** — grade 2 or worse. It means "this person needs to see a specialist". This is the
decision that actually matters in real screening.

**QWK (Quadratic Weighted Kappa)** — a score for how well my model's grades match the real
grades. 0 means my model is guessing randomly. 1 means perfect. The useful part is that it
punishes big mistakes more than small ones. Saying grade 4 when the truth is grade 0 hurts my
score much more than saying grade 1.

**AUROC** — a score for how well my model separates "needs referral" from "doesn't". 0.5 means
it is a coin flip. 1.0 means perfect. Mine is 0.887.

**Sensitivity** — out of all the people who really needed referral, what fraction did I catch.

**Specificity** — out of all the healthy people, what fraction did I correctly leave alone.

**Confidence interval (CI)** — a range instead of a single number. It means "the real answer is
probably somewhere in here". This matters because if the range includes zero, then whatever
effect I measured might actually be nothing at all. I put a confidence interval on every single
number in my project.

**Train/test split** — you train the model on some photos and test it on different photos it has
never seen. Otherwise you are just checking whether it memorised.

**Patient-level split** — my way of splitting. Both of a person's eyes always go to the same
side. Either both are used for training, or both are used for testing, never one of each.

**Image-level split** — the sloppy way. You just shuffle all the photos. This means a person's
left eye can end up in training while their right eye ends up in testing.

**Backbone** — the big pre-built image network I started from instead of building one from
scratch. I used EfficientNet-B0 and ResNet-50.

**Fine-tuning** — training the whole network end to end on my eye photos.

**Frozen features** — a cheaper shortcut. I run every photo through the network once, save the
numbers it produces, and then only train a tiny classifier on top of those saved numbers. It
takes seconds instead of hours, so I could run lots of experiments.

**Classifier head** — the last small piece of the model that turns the network's output into an
actual grade.

**Multinomial head** — treats the five grades as five unrelated categories, like five different
animals.

**Ordinal head** — treats the grades as being in order, because they are. Grade 3 really is
worse than grade 2. This turned out to matter a lot.

**Class imbalance** — most of my photos are healthy eyes. About 73%. So the model can score well
just by saying "healthy" all the time, which is useless.

**Bootstrap** — a way of finding out how much a number wobbles. I randomly re-pick my test
patients over and over and recalculate. If the number jumps around a lot, I should not trust it.

**Epoch** — one full pass of the model through all the training photos. More epochs means more
learning, up to a point.

**Early stopping** — the model stops training by itself when it stops improving, instead of
running a fixed number of epochs and hoping that was enough.

---

## Part 2 — The dataset and what I did with it

**1. Two public datasets, very different sizes.**
EyePACS gives me 35,126 photos from 17,563 patients, and every patient has both eyes
photographed. APTOS 2019 gives me 3,662 photos with no eye pairing at all. Together that is
38,788 rows. EyePACS does all the real work; APTOS is only used where eye pairing is not needed.

**2. The data is heavily imbalanced.**
The grade counts are 25,810 / 2,443 / 5,292 / 873 / 708. About 73% of images are healthy eyes,
and the two most severe grades together are under 1,600. A model could score reasonably just by
saying "healthy" every time, which is useless for screening.

**3. Neither dataset tells you who the patient is — so I rebuilt that.**
This was the biggest data job. EyePACS hides the patient ID and which eye it is inside the
filename. I parsed it out, validated it, and built a proper table with a `patient_id` and a
`partner_id` (which photo is the other eye) on all 38,788 rows. Without this, patient-level
splitting and the whole both-eyes experiment would have been impossible.

**4. I checked my table was actually correct.**
It is easy to build a table that silently misreads the source. So I compared my grade counts
against the officially published totals for EyePACS and they matched exactly. That is the check
that told me I was reading the data properly before I built anything on top of it.

**5. I found 8 photos that were completely black.**
Failed captures, not real photographs. I caught them with a check on average pixel brightness
after preprocessing. I excluded them before splitting the data, and verified where each one had
ended up — only one reached a test set at all, and zero reached my main one. I flagged them in
the table rather than deleting the rows, so the evidence is still there. Three of them had been
graded "1" by a human, which is someone labelling a black rectangle as mild disease.

**6. I found 148 pairs of near-duplicate photos.**
I used perceptual hashing — it turns each image into a fingerprint so near-identical photos can
be matched. All 148 pairs were inside APTOS, across 131 groups and 270 images. None in EyePACS,
none across the two datasets. If I had left them in, near-identical photos could land on both
sides of a split and inflate my scores through a completely different route than the one I was
watching for.

**7. Then I went back and audited my own duplicate setting.**
I had used a strict threshold and later worried it was too strict and missing real duplicates.
So I tested it instead of guessing: I made a fake near-duplicate on purpose and confirmed it
scored 0 distance, swept the threshold across all 616,900,375 possible EyePACS pairs, and
examined the 790 closest candidates. 786 of them were different patients and their grades agreed
barely above chance (62.7% against a 56.8% baseline) — meaning they are coincidental lookalikes
from the same camera setup, not real duplicates. So I left the setting alone and wrote a test
that fails loudly if this ever stops being true.

**8. I built three ways of splitting the data and locked the test set.**
P2 splits by patient (both eyes stay together) and is my honest protocol. P1 splits by image on
purpose so I could measure how bad that is. P3 trains on one dataset and tests on the other.
Twenty automatic tests check all of it — seven on the splits, eight on the table, five on the
duplicate detection. Then I declared the test sets at a specific point in my code history, before
any model had been tested on anything, and only scored them at the end.

---

## Part 3 — The report, section by section

### Abstract
This is the summary at the top. It says what the disease is, what I built, what score I got, and
what the three main things I learned were. If someone only reads one paragraph, this is it.

### 1.1 Problem Statement
I explain what diabetic retinopathy is and why automatic grading is needed. The important point
I make here is that missing a sick person is much worse than accidentally flagging a healthy
person. That is why I judge my model on the referral decision, not on getting the exact grade
right.

I also explain one problem I ran into early. Some published papers report accuracy that is
higher than two human doctors agree with each other. That cannot really be true. A model cannot
be more consistent than the answers it was taught from. So I decided not to trust published
numbers as my target and instead build my own way of measuring things.

### 1.2 Literature Review
Twelve papers. For each one I say what I actually took from it and used, not just what it said.

The main ones:
- **Gulshan (2016)** got 0.99 AUC but used 128,000 private photos that multiple doctors graded.
  I cannot get that data. So I use this as the ceiling, not as my target.
- **Voets (2019)** tried to redo Gulshan's work with only public data and got 0.951 and 0.853.
  This is the fairest comparison for me, because we are both using public data.
- **Gargeya (2017)** used a pretrained network plus a simple classifier on top. I copied that
  general design because it is cheap to run.
- **Tymchenko (2020)** used an ordinal head in a Kaggle competition. I borrowed that idea and it
  became my biggest improvement.
- **Ali (2025)** built a model that uses both eyes and said it helps — but never checked *why*
  it helps. That gap is exactly what my own experiment fills.
- Three papers about **data leakage** (Tampu, Yagis, Rouzrokh) taught me to check my own splits.

### 1.3 Challenges
Seven problems I hit, in the order I hit them.

1. **Class imbalance.** Most eyes are healthy. I used the standard fix, and it turned out to be
   the wrong call — see section 3.3.
2. **Grade 1 might be impossible.** Grade 1 is defined by tiny spots about 10 pixels wide. When
   I shrink photos to 224 pixels for the model, those spots basically disappear.
3. **Compute limits.** I had one laptop GPU. My first training run stopped too early and the
   model was still improving. I had to notice that and redo it properly. Longest job took five
   and a half hours.
4. **No patient IDs.** Neither dataset tells you which photos belong to the same person. I had
   to dig that out of the filenames myself and build a proper table of 38,788 rows.
5. **Broken data.** I found 8 photos that were completely black. And 148 pairs of near-identical
   duplicate photos in one dataset. Both would have quietly ruined my results.
6. **I audited my own duplicate-finder.** I had already used a setting and later worried it was
   too strict. So I tested it properly across 616 million photo pairs. It turned out my original
   setting was fine, so I left it alone and wrote a test to lock the decision in.
7. **I found a mistake in my own experiment.** Explained in section 3.4.

### 2.1 Objectives
Five goals. Build a working model, find out what makes it work, test whether using both eyes
helps, find out where it fails, and make sure my measurements are trustworthy.

### 2.2 Methodology
How I set everything up.

- **2.2.1** — The two datasets, and the cleaning I did first.
- **2.2.2** — My three ways of splitting the data. P2 is the honest one and all my headline
  numbers come from it. P1 is the sloppy one, built on purpose so I could measure how bad it is.
  P3 trains on one dataset and tests on the other.
- **2.2.3** — I locked my test data at a specific point in my code history, before any model had
  been tested. This stops me from cheating by trying things until something looks good.
- **2.2.4** — My one rule: every number gets a confidence interval, and every comparison between
  two things gets its own interval on the difference. If that interval includes zero, I do not
  claim the difference is real.
- **2.2.5** — The models and the two classifier heads.
- **2.2.6** — How I set up the both-eyes experiment so I could separate three things instead of
  changing them all at once.
- **2.2.7** — Practical stuff: every training job can be stopped and resumed, and I log the score
  after every epoch so I can see whether the model actually finished learning.

### 3.1 What I Built
The size of the thing. 7,179 lines of code, 28 files, 8 test files, about 40 trained models, 20
automatic tests that check my data is correct.

### 3.2 The Screening Model — **this is my main result**
I trained it twice.

The first time I gave it 15 epochs and it was still getting better when I stopped it. I could
tell because my cheap shortcut model was beating my expensive model, which should never happen.
So I trained again properly: 40 epochs with automatic stopping. All 8 runs stopped on their own
between epoch 18 and 35, which means they genuinely finished learning.

Final numbers: **QWK 0.671, AUROC 0.887** on the honest patient-level split.

Then the important part. A screening model is never used at its default setting. You pick a
cut-off based on how many sick people you need to catch. So I set mine to catch 90% of referable
patients, and at that setting it correctly clears 61.3% of healthy people. It misses 124 out of
1,246 referable patients.

I am honest that 61.3% is not good enough. It means about 2 in 5 healthy people would be sent to
a specialist for nothing.

### 3.3 Which Design Choices Mattered
This is where I say what actually made the model better.

- **The ordinal head was the biggest win.** Just changing how the model treats the grades — as
  ordered instead of unrelated — gave +0.05 to +0.09. It works on all three networks I tried and
  costs nothing.
- **Higher resolution helped** (+0.058 going from 224 to 384 pixels), which makes sense given
  the tiny lesions.
- **The class balancing made things worse.** This surprised me. I turned it off and the model
  improved in all four runs. And the thing it was supposed to help — catching grade 1 — did not
  change at all. So I was paying a price for nothing.

The lesson I would give someone else: test your defaults instead of assuming they help.

### 3.4 Using Both Eyes
My first result said using both eyes gave +0.075. It looked great.

Then I looked at what I had actually changed between the two things I compared, and I had
changed three things at once: the head, how the eyes get combined, and how the two grades get
merged. So I could not honestly say the improvement came from using both eyes.

Nobody told me this. I found it myself.

So I built a proper experiment that changes one thing at a time. The answer: about two-thirds of
that +0.075 was actually the ordinal head, not the second eye. Using both eyes does help a bit
(+0.025), but on one of my three networks it actually made things *worse*.

So my honest conclusion is smaller than my first one: always use the ordinal head, and only add
the both-eyes trick after checking it works on your specific network.

### 3.5 Where the Model Breaks Down
Two honest limits.

**Grade 1.** My model only catches 8.7% to 14% of grade 1 cases. But three separate things point
at the same explanation. Grade 1 is also the only grade that does not match between a person's
two eyes. And 3 of the 8 completely black photos I threw away had been graded as grade 1 by a
human — someone labelled a black rectangle as "mild disease". So grade 1 might not be something
my model is failing at. It might be something the photos genuinely cannot show at this size.

**Moving between datasets.** Training on the big dataset and testing on the small one works okay
(0.754). The other direction falls apart (0.287). Mostly because the small dataset is just too
small to learn from.

### 3.6 Was My Evaluation Honest?
This section checks my own work.

First I measured how similar a person's two eyes are. Very similar: 0.855.

That leads to my favourite thing I built. If I just copy one eye's grade onto the other eye — no
photo, no model, nothing — I get 0.838. So any model scoring near 0.838 might be cheating off
that similarity instead of actually reading the photo. All my models score well below it, which
is good.

Then the big test: does the sloppy image-level split actually inflate scores like the papers
warn? I predicted yes. **The answer was no.** I tested it three different ways and got nothing
every time. I report that honestly instead of hiding it.

### 3.7 Discussion
What all of it means together, including where I disagree with a published paper and why.

### 3.8 What I Predicted vs What Happened
A table of everything I wrote down before running experiments, next to what actually happened.
I got more of them wrong than right. I left this in on purpose — it shows I wrote my predictions
down first instead of deciding afterwards.

### 4 and 5 — Future Work and Conclusion
See Part 5 below for the full list of what is left.

---

## Part 4 — The presentation, slide by slide

What I would actually say out loud on each slide. Roughly 1 to 2 minutes each.

**1. Title** — "This project is about building a model that reads eye photos and decides who
needs to see a specialist."

**2. The clinical problem** — Explain the disease, why early detection matters, and that there
are not enough graders. Key line: missing a sick person is much worse than flagging a healthy
one.

**3. What I built** — The scale of the work. 7,179 lines of code, 40 models, all on one laptop.

**4. Data and grading scale** — The two datasets, the 0-4 scale, and the cleaning I did.

**5. The imbalance chart** *(new slide)* — Show that most eyes are healthy. Point at grade 1 and
say "remember this bar, it comes back twice."

**6. How I set up the evaluation** — Three splits, locked test set, confidence interval on
everything.

**7. The model** — Two training passes. Why the first one was not finished, how I noticed, and
the final numbers.

**8. The clinical operating point** — My main result. The ROC curve. "At 90% sensitivity I get
61.3% specificity." Be honest that this is not deployable yet.

**9. Comparison to published work** — My 0.887 sits between the two public-data results. The gap
to 0.99 is about data, not about my model being bad.

**10. The ordinal head** — My biggest win, and it was free.

**11. The sampler I was wrong about** — Turning off the standard fix made things better. Say
plainly: "I assumed this was helping and it was not."

**12. The mistake I caught** — The +0.075 that looked great, and why I could not report it.

**13. The decomposition** — What was really causing the improvement. Two-thirds was the head.

**14. Where it breaks down** — Grade 1 and the three clues pointing at the same cause.

**15. Cross-dataset transfer** *(new slide)* — The 0.754 vs 0.287 asymmetry.

**16. Was my evaluation honest?** — The 0.838 shortcut ceiling and the leakage tests that came
back empty.

**17. Every effect on one axis** *(new slide)* — The forest plot. Say: "the head is the only
thing that works everywhere; everything else is small or uncertain."

**18. What I got wrong, and what's next** — The prediction table and the remaining work. End on:
"I needed proof before calling anything real, including for my own predictions, which is why
three of them are on this slide instead of deleted."

---

## Part 5 — What is left to do

**1. Calibration — make the confidence scores mean something.**
Right now my model outputs a number that acts like a probability but is not really one. Modern
networks are usually overconfident. I will fit a single scaling value on validation data (never
on test) and check it with reliability diagrams. This is the highest priority, because my model
has to run at a threshold far from its default — and a threshold is only as trustworthy as the
probabilities underneath it.

**2. A "not sure" option.**
Once the probabilities are honest, the model can refuse to answer and hand the hardest cases to
a human grader instead of guessing. I will compare two ways of measuring uncertainty, sweep the
cut-off, and pick it by a rule I fix *in advance* — the lowest threshold where validation
sensitivity still hits 90% — rather than by looking at test results afterwards. This is also the
most likely route to fixing my weak 61.3% specificity.

**3. Settle whether grade 1 is actually learnable.**
I will merge grades 0 and 1 into one class, retrain, and see what happens to everything else.
The trick is scoring both models in the *same* label space so the comparison is fair, and using
referable sensitivity as a fixed reference point since it does not change when you merge 0 and
1. If the score jumps while referral performance holds, grade 1 was costing me accuracy and
buying nothing. Code is written.

**4. Explain why my leakage result disagrees with the published one.**
My best guess is dataset size — the OCT study that found leakage used small datasets, mine has
27,145 training images. I will shrink my training set to 500, 1k, 2k, 5k, 10k and 20k images and
plot how the gap changes. If it opens up at small sizes, my result becomes a *boundary
condition* on theirs instead of a contradiction. If it stays flat, I have ruled out the obvious
explanation and my result gets stronger. Code is written.

**5. Finish the two leftover checks on the converged models.**
The leakage tests ran on frozen features and on an undertrained model — the setup least able to
memorise anything. I need to re-run both on the properly converged models, and report the
smallest effect the test could have detected, because a null result is only as strong as what it
rules out. Code is written; I have already dumped the predictions it needs.

**6. Look at what the model is actually attending to.**
Grad-CAM produces a heatmap over the photo showing which regions drove the decision. My rule is
that the panel has to include cases the model got *wrong*, not just a handpicked set of correct
ones — otherwise it only confirms what I already believe.

**7. Build the actual screening app.**
A simple interface where you upload a photo and get back a grade in plain language, a
referable/not-referable decision, a confidence score or an "uncertain — refer" flag, and the
heatmap. It has to import my own preprocessing code rather than reimplement it, which is a small
mistake that commonly breaks deployed models. Before I call it done, 100 images go through both
the app and my evaluation pipeline and the answers must match exactly.

**8. Survey how other people split their data, then write the final paper.**
Check 20 public GitHub projects that train DR models on these datasets and record whether each
splits by image or by patient. I will report only the overall fraction, not name anyone. And I
will frame it carefully: my own finding is that image-level splitting did *not* measurably hurt
me, so the point is that the practice is common and probably less harmful than assumed — not
that the field is doing it wrong.

---

## Part 6 — 25 questions my teacher might ask, and my answers

### The basics

**1. What is diabetic retinopathy and why does it need a computer model?**
It is damage to the blood vessels at the back of the eye caused by diabetes. It is one of the
biggest causes of preventable blindness in working-age adults. The reason it is dangerous is
that it does not hurt and does not blur your vision until it is already advanced, so people do
not know to go to a doctor. That is why diabetics get a yearly eye photo taken as a check-up.
The problem is volume: there are far more photos taken each year than there are trained
specialists to look at them, and diabetes is getting more common. A computer model that can
handle the easy, obviously-healthy cases would let the specialists spend their time on the cases
that actually need a human.

**2. What exactly does your model do?**
You give it one photo of the back of an eye. It gives back two things. First, a grade from 0 to
4 saying how advanced the disease looks. Second, and more importantly, a yes/no on whether this
person should be sent to a specialist — which is grade 2 or above. It does not diagnose and it
does not treat. It sorts.

**3. Why is the referral decision more important than getting the exact grade right?**
Because that is the decision that changes what happens to the patient. Whether someone is graded
3 or 4 does not change anything on the day — either way they are going to a specialist. But
whether they are graded 1 or 2 decides whether they get sent at all. And the two mistakes are
not equally bad. If I wrongly flag a healthy person, they waste an appointment. If I miss a sick
person, they can go blind. So I judge my model on the referral decision and I set it up to be
cautious in the direction that matters.

**4. Why did you use public datasets instead of collecting your own?**
I do not have access to a hospital or to patients, and getting ethical approval to collect
medical images is not something I could do for a project at this scale. There are two good
public datasets — EyePACS and APTOS — and using them means anyone can check my work by
downloading the same data. The downside, which I am upfront about, is that public data is graded
by a single person per image rather than by several doctors who then agree on an answer. That
puts a ceiling on how good any model trained on it can be, and I show exactly where that ceiling
is when I compare against published work.

### The data

**5. Tell me about your datasets.**
EyePACS is the main one: 35,126 photos from 17,563 patients. Every patient has both eyes
photographed, which is the property my whole both-eyes experiment depends on. APTOS 2019 is much
smaller, 3,662 photos, and it does not tell you which photos came from the same person at all.
So APTOS only gets used in places where I do not need to know that. Together they are 38,788
rows in my table. The data is very imbalanced — 25,810 of the EyePACS photos are healthy eyes,
which is about 73%.

**6. What was wrong with the raw data and how did you fix it?**
Two things, and neither of them shows up when you just train a model — you have to go looking.
First, eight photos were almost completely black. They were failed captures where the camera did
not fire properly. I found them by checking the average brightness of every image after
processing, and I removed them before splitting the data. Second, I found 148 pairs of
near-identical photos inside APTOS using perceptual hashing, which turns each image into a short
fingerprint so you can find lookalikes. If I had left those in, the same photo could effectively
appear in both training and testing, which would have made my results look better than they are.

**7. What is a patient-level split and why does it matter?**
When you train a model you have to hold back some data to test on. The lazy way is to shuffle
all the photos randomly. The problem is that a person has two eyes, and both eyes have the same
disease, so they look similar. If you shuffle randomly, someone's left eye can go into training
while their right eye goes into testing. The model then has a hint about the test photo that it
should not have. My patient-level split keeps both of a person's eyes on the same side, always.
That is stricter and gives a more honest number.

**8. How did you know your patient IDs were correct?**
This was the part I was most worried about getting silently wrong, because everything else is
built on it. EyePACS hides the patient ID and which eye it is in the filename, so I had to parse
that out myself. To check I had done it right, I counted how many photos of each grade my table
contained and compared that against the grade counts EyePACS officially publishes. They matched
exactly — 25,810, 2,443, 5,292, 873, 708. If I had been misreading the files, those totals would
not have lined up. I also wrote 20 automatic tests that re-check the table and the splits every
time I run them.

**9. How did you handle the class imbalance?**
Initially with the textbook fix: class-balanced resampling, which means showing the model the
rare grades more often during training so it does not just learn to say "healthy" every time.
But I made it a switch I could turn on and off rather than a fixed setting, and that turned out
to be important — when I actually tested it, turning it off made the model better in all four
runs. I talk about that more in question 19.

### The method

**10. Which model architecture did you use and why?**
EfficientNet-B0 is my main one. I picked it because it gives good accuracy for its size, which
mattered a lot since everything ran on a single laptop GPU. I also tested ResNet-50, which is
older and more standard, so I could check whether my findings were specific to one architecture
or general. I did not design a network from scratch — I started from networks already trained on
millions of general images and adapted them to eye photos, which is the standard approach when
you have tens of thousands of images rather than millions.

**11. What is the difference between frozen features and fine-tuning, and why did you use both?**
Fine-tuning means training the whole network on my eye photos. It gives the best result but each
run takes hours. Frozen features means I push every photo through the network once, save the
numbers that come out, and then only train a small classifier on top of those saved numbers.
That takes seconds instead of hours. I used frozen features for all my comparison experiments,
because I needed to run dozens of them — three networks, five random seeds, ten different
configurations. Then I used full fine-tuning to produce the actual final model. Without the
frozen-feature shortcut I could not have run the experiments at all on the hardware I had.

**12. What is an ordinal head and why did it help so much?**
The head is the last little piece of the model that turns the network's output into a grade. The
normal way is a softmax, which treats the five grades as five unrelated labels — like it is
choosing between cat, dog, horse, bird, fish. But grades are not unrelated. Grade 4 is worse
than grade 3 which is worse than grade 2. They are in order. My ordinal head predicts a single
number on a continuous scale and then cuts it into five bands using four thresholds that I
optimise on the training data. That way, if the truth is grade 4 and the model says 3, it is
only slightly wrong, and the model learns that. This gave me +0.05 to +0.09 on every network I
tried, and it costs essentially nothing to implement. It was the single biggest improvement in
my whole project.

**13. How did you know your model had finished training?**
This is a mistake I made and then caught. My first training run had a fixed budget of 15 epochs.
When it finished, I noticed something odd: my cheap frozen-feature shortcut model was scoring
0.621 and my expensive fully-trained model was only scoring 0.555. A properly trained model
should never lose to a cut-down version of itself. That told me it had simply not finished
learning. I checked the per-epoch logs and confirmed it — the best epoch was epoch 11 or later
in four out of five runs, meaning it was still improving when I cut it off. So I retrained with
40 epochs and automatic early stopping. All eight runs stopped on their own between epoch 18 and
35, well before the limit, which is what genuinely finished training looks like. And the final
score went to 0.671, which now beats the shortcut as it should.

**14. Why did you test three different backbones instead of just one?**
Because otherwise I would not know whether a finding is about my data or about one particular
network. This turned out to matter a lot. My both-eyes result was positive on two networks and
actually *negative* on the third. If I had only run EfficientNet-B0, I would have confidently
written that using both eyes helps, full stop. Running three is what let me say the honest
version: it helps on some architectures and hurts on others, so check it on yours.

**15. What is QWK and why did you use it?**
Quadratic Weighted Kappa. It measures how well my predicted grades agree with the real grades,
where 0 is random guessing and 1 is perfect. The reason it is the right metric here rather than
plain accuracy is that it weights mistakes by how far off they are. If the truth is grade 0 and
I say grade 4, that is a much bigger error than saying grade 1, and QWK reflects that. Plain
accuracy would treat both as simply "wrong". It is also the standard metric for this task, so my
numbers can be compared to other people's.

**16. What is a confidence interval and why is it on every number?**
A confidence interval is a range rather than a single number — it says "the true value is
probably somewhere in here". I use it because a single number hides how much luck was involved.
If I say my model improved by 0.02, that means nothing until you know whether the range around
it is [0.01, 0.03] or [-0.05, 0.09]. In the second case the improvement could easily be zero.
I calculate mine by bootstrapping: I randomly re-pick my test patients thousands of times and
recalculate the score each time, then take the middle 95% of the results. Crucially I re-pick
*patients*, not photos, so a person's two eyes always move together — otherwise the range would
come out falsely narrow. And I never compare two numbers by eye; I calculate a separate interval
on the difference between them, and only call it real if that interval excludes zero.

### The results

**17. What is your final result?**
On my honest patient-level test set, the model gets QWK 0.671 for the five-grade task and AUROC
0.887 for the referral decision. AUROC 0.887 means that if you give it one sick person and one
healthy person, it ranks the sick one as more likely to need referral about 89% of the time. At
the operating point where it catches 90% of referable patients, it correctly clears 61.3% of
healthy people. It misses 124 out of 1,246 referable patients at that setting.

**18. Why is your score lower than the published papers?**
Mostly because of data, and I can show that rather than just claim it. Gulshan's 0.99 came from
about 128,000 private photos where several ophthalmologists graded each image and then resolved
disagreements. That quality of labelling is not available publicly. The fairest comparison is
Voets, who deliberately tried to reproduce Gulshan's work using only public data — the same
situation I am in — and got 0.951 and 0.853 on their two test sets. My 0.887 sits right between
those two. So my result is exactly where a public-data result should land. The gap to 0.99 is a
data gap, not evidence that my model is badly built.

**19. Your specificity is 61% — isn't that too low to actually use?**
Yes, and I say so in the report rather than hiding it. It means roughly two in every five
healthy people would be sent to a specialist unnecessarily. I would not deploy that. But I can
name three specific reasons for it rather than shrugging. First, my model predicts five grades
and I convert that into a yes/no afterwards, whereas Voets trained directly for the yes/no
decision — training for the thing you are measured on usually wins. Second, I used 224-pixel
images, and my own test showed that going to 384 pixels gains several points, so resolution is
costing me. Third, my test set is patient-level, which is stricter than the image-level split
most published numbers use. The first two are fixable with more compute than I had, and the
calibration and reject-option work in my future plan targets exactly this.

**20. What did you find about using both eyes?**
It helps, but much less than my first result suggested, and I only found that out by checking my
own work. My initial comparison showed +0.075, which looked great. Then I realised I had changed
three things at once between the two conditions I was comparing: the classifier head, how the
two eyes get combined, and how the two grades get merged into one. So I could not honestly
attribute the gain to the second eye. I built a proper experiment that varies one thing at a
time. It turned out about two-thirds of that improvement was actually the ordinal head, which
has nothing to do with having two eyes. The genuine both-eyes benefit is about +0.025 — real but
small — and on one of my three networks it was actually negative.

**21. Why can't your model detect grade 1?**
It catches only 8.7% to 14% of grade 1 cases, which is well below what I expected. But I do not
think this is simply my model being bad, because three completely separate findings all point at
the same cause. Grade 1 is defined by microaneurysms, which are tiny dots about 10 pixels wide
in the original photo — and I shrink photos to 224 pixels, so that evidence is largely destroyed.
Second, grade 1 is the only grade that does not match between a person's two eyes: 45.8% of the
time, versus 72% to 94% for every other grade. If even the disease itself does not agree between
two eyes of the same person, the signal is very weak. Third, three of the eight completely black
photos I removed had been graded as "1" by a human — someone labelled a black rectangle as mild
disease, which tells you human graders are also unreliable on this class. Taken together, grade
1 may be a limit of the data rather than a limit of my model. I have written an experiment to
test that directly.

### Rigour and honesty

**22. You predicted that leakage would inflate scores and it didn't. Isn't that a failed
experiment?**
No, and this is a distinction I care about. Before running it I wrote down in my plan that "no
difference" was an acceptable outcome. That matters, because if you only accept one answer you
are not really testing anything. I then checked it three separate ways — comparing the two split
protocols, running a targeted test on which patients had their partner eye in training, and
checking it again under full training — and got nothing every time. A result that disagrees with
my expectation is still a result. What makes it interesting rather than boring is that it
disagrees with a published finding in a very similar field, retinal OCT scans, where the effect
was clearly present. I have designed an experiment to explain that difference: my best guess is
that it is about dataset size, since their datasets were small and mine has 27,145 training
images, and a model with plenty of real signal has no reason to fall back on memorising patients.

**23. How do you know your numbers are not just luck?**
Three things. Every number has a bootstrap confidence interval, so I can see how much it wobbles
when the test patients change. Every comparison is done as an interval on the *difference*, not
by eyeballing two numbers, and I only call a difference real if that interval excludes zero. And
I ran multiple random seeds for the things that depend on randomness, so I can see run-to-run
variation directly. There is also a fourth safeguard: I locked my test set at a fixed point in
my code history before any model had ever been scored on it, which stops me from unconsciously
trying things until something looks good.

**24. What mistake did you find in your own work?**
Two, and I kept both in the report. The first was the both-eyes comparison in question 20, where
I had changed three variables at once and nearly reported a number I could not justify. The
second was the class-balancing setting. I had turned it on because that is what you are supposed
to do with imbalanced data, and I assumed it was helping. When I finally made it an experiment
instead of a fixed setting, it turned out to be making the model worse in every single run — and
the grade-1 recall it was supposed to be protecting did not change at all. So I had been paying
a cost for a benefit that was never arriving. The general lesson I took from both is to turn
your assumptions into experiments.

**25. What would you do differently, and what is left?**
Differently: I would test my defaults much earlier. The class-balancing setting was wrong for
weeks before I checked it, and I only found out because I turned it into an ablation. I would
also have started at higher resolution, since 384 pixels clearly helps and the disease is
defined by small features. What is left, in priority order: calibration so the model's
confidence scores are trustworthy, then a "not sure — send to a human" option, which together
are the most likely fix for my weak specificity. Then the grade-1 experiment to settle whether
that class is learnable at all, the dataset-size experiment to explain my disagreement with the
published leakage result, Grad-CAM heatmaps to see what the model is actually looking at
including on its mistakes, and finally an actual working app. The code for four of those is
already written.
