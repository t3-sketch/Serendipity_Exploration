# Exploration of Serendipity

### Finding Undiscovered Bangers with a Fame-Agnostic Audio Model

By **Taiyo Morita** — A project for DSC 80 at UCSD

---

## Introduction

Why do some incredible songs stay buried while objectively similar tracks rack up millions of streams? Streaming platforms tend to recommend what is _already_ popular, reinforcing a filter bubble in which an artist's existing fame, not the music itself, drives discovery. This project asks a deliberately fame-blind question:

> **Can a track's popularity be predicted solely from its intrinsic audio characteristics, completely independent of the artist's existing fame? Furthermore, can the residuals from this fame-agnostic model be combined with a fame penalty to systematically unearth "Undiscovered Bangers" — tracks that possess the sonic architecture of a hit but remain buried due to low exposure?\*\***

The idea is simple but powerful. If a model that knows nothing about artist fame still predicts that a track _should_ be popular, yet its real popularity is low, that track is a serendipity candidate: musically strong but under-exposed.

The analysis uses the **Spotify Music Tracks** dataset. The raw `music_tracks.csv` contains **114,000 rows** and `artists.csv` contains **1,162,095 rows**. After filtering to six genres and merging the two files, the working dataset has **4,797 rows**, one per track.

We focus on six musically distinct genres — **classical, jazz, k-pop, hip-hop, anime, and j-dance** — chosen to span a wide range of acoustic profiles (from low-energy acoustic classical to high-energy electronic j-dance).

The columns most relevant to our question are:
**`music_tracks.csv`**

| Column             | Description                                           |
| ------------------ | ----------------------------------------------------- |
| `popularity`       | Target variable. Spotify popularity score from 0–100. |
| `danceability`     | How suitable a track is for dancing (0–1).            |
| `energy`           | Perceptual intensity and activity (0–1).              |
| `loudness`         | Overall loudness in decibels (dB).                    |
| `speechiness`      | Presence of spoken words (0–1).                       |
| `acousticness`     | Confidence the track is acoustic (0–1).               |
| `instrumentalness` | Likelihood the track has no vocals (0–1).             |
| `liveness`         | Presence of a live audience (0–1).                    |
| `valence`          | Musical positivity / mood (0–1).                      |
| `tempo`            | Estimated tempo in beats per minute (BPM).            |
| `explicit`         | Whether the track has explicit lyrics (boolean).      |
| `track_genre`      | Genre label for the track.                            |

**`artists.csv`**

| Column                     | Description                                                                                                                    |
| -------------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| `name`                     | the name of the artist                                                                                                         |
| `followers` → `fame_score` | Artist follower count, transformed into a 0–1 fame score. Deliberately excluded from the model so predictions stay fame-blind. |

---

## Data Cleaning and Exploratory Data Analysis

### Data Cleaning

Our cleaning pipeline was designed around the data generating process (DGP) of a streaming catalog, where the same song appears many times and artist metadata lives in a separate file.

1. **Tempo recovery before deduplication.** Many tracks appear as multiple releases (remasters, bonus editions). Because some of these duplicates carried a valid `tempo` while others were missing it, we first forward/back-filled `tempo` within each duplicate group so we would not throw away usable values when collapsing duplicates.
2. **Deduplicating releases.** We sorted by `popularity` (descending) and kept one representative row per `(track_name, artists)` pair, so the surviving record reflects the track's best-performing release. Ties were broken by keeping the first occurrence.
3. **Resolving collaborations and merging artists.** `music_tracks.csv` has no artist ID, so the only way to attach follower counts was to join on artist _name_. We split semicolon-separated collaborators with `.str.split(';')`, `.explode()`-ed them into individual rows, left-joined `artists.csv` (keeping the highest-follower entry for duplicated names), and then retained, for each `track_id`, the single collaborator with the most followers. Tracks whose artist had no match in `artists.csv` were left with `NaN` followers rather than dropped.
4. **Removing system anomalies.** We dropped one fully corrupted record whose `duration_ms` was 0 with `NaN` artist and track name.
5. **Engineering `fame_score`.** We log-transformed `followers` with `log(x + 1)` to tame its extreme long-tail distribution, then applied a `MinMaxScaler` to map it onto a 0–1 `fame_score`. This column is **never given to the predictive model**, since encoding artist fame is exactly the filter-bubble effect we want to avoid. We utilize this for the downstream serendipity calculation, where it functions as a critical fame penalty weight to surface truly overlooked tracks.

**A note on a known limitation:** because deduplication matches on the exact `track_name` string, it cannot collapse _version variants_ of the same underlying song (e.g. a studio recording vs. its "- LIVE" or remastered counterpart, or a track listed under both its original and translated title). As a result, multiple renditions of a single song can persist in the data, and the serendipity showcase can surface more than one version of the same track.

The head of the cleaned DataFrame:

| track_id               | artists       | track_name         | popularity |  duration_ms | explicit | danceability | energy | key | loudness | mode | speechiness | acousticness | instrumentalness | liveness | valence |   tempo | time_signature | track_genre | fame_score | popularity_tier |
| :--------------------- | :------------ | :----------------- | ---------: | -----------: | :------- | -----------: | -----: | --: | -------: | ---: | ----------: | -----------: | ---------------: | -------: | ------: | ------: | -------------: | :---------- | ---------: | :-------------- |
| 000RDCYioLteXcutOjeweY | Jordan Sandhu | Teeje Week         |         62 |       190203 | False    |        0.679 |   0.77 |   0 |   -3.537 |    1 |        0.19 |       0.0583 |                0 |   0.0825 |   0.839 | 161.721 |              4 | hip-hop     |   0.691528 | High (51–75)    |
| 005NLlv3rClEVqeoSyYTHU | Leola         | キミが好きで、、、 |         20 |       281333 | False    |        0.797 |  0.667 |   7 |   -3.975 |    1 |      0.0382 |        0.473 |                0 |   0.0908 |   0.286 | 120.036 |              4 | j-dance     |   0.508147 | Low (0–25)      |
| 006tmNZLXEXPqdb23wwSN1 | İlhan İrem    | Yemyeşil Bir Deniz |         44 |       358173 | False    |        0.486 |  0.568 |   9 |   -9.199 |    0 |      0.0417 |        0.652 |                0 |    0.834 |    0.65 |     nan |              4 | jazz        |   0.587311 | Mid (26–50)     |
| 00Coyxt9mTec1acC52qtWa | TAEIL         | Starlight          |         62 |       225817 | False    |        0.643 |  0.735 |   2 |   -3.703 |    1 |      0.0302 |       0.0786 |                0 |    0.101 |   0.502 |     nan |              4 | k-pop       |   0.614997 | High (51–75)    |
| 00EAiVX99rZ4rlYFLItPcC | はてな        | 声?                |         52 |       247306 | False    |         0.42 |  0.964 |   5 |    -3.35 |    1 |      0.0902 |        0.015 |                0 |    0.207 |   0.584 | 170.019 |              4 | anime       |   0.370915 | High (51–75)    |
| track_genre            | popularity    | danceability       |     energy | acousticness | explicit |      valence |

### Univariate Analysis

Track popularity is heavily right-skewed, with a pronounced spike at 0 representing tracks with virtually no streaming activity. Beyond this zero-popularity mass, the distribution exhibits a distinct local peak emerges around 20, followed by a broader concentration between 40–70. This suggests that tracks are stratified into three tiers: largely undiscovered, emerging, or moderately mainstream.

<iframe
  src="assets/pop-dist.html"
  width="800"
  height="500"
  frameborder="0"
></iframe>

Tempo follows an approximately normal distribution centered around 100–120 BPM, consistent with the dominance of dance-oriented genres (k-pop, j-dance, hip-hop) in our selection. The long right tail extending past 150 BPM reflects the high-energy anime and j-dance tracks.

<iframe
  src="assets/tempo-dist.html"
  width="800"
  height="500"
  frameborder="0"
></iframe>

### Bivariate Analysis

A clear positive association is visible between an artist's fame score and a track's popularity, confirming that fame is a strong confound the model must deliberately ignore. Crucially, the top-left zone — high popularity despite a low fame score — is exactly where the serendipity candidates this project targets live.

<iframe
  src="assets/fame-scatter.html"
  width="800"
  height="600"
></iframe>

Higher popularity tiers tend to exhibit higher median danceability, with the Very High (76–100) tier clustering around 0.75 versus roughly 0.55 for the Low tier. However, the wide interquartile ranges across all tiers suggest that danceability alone is far from sufficient to predict popularity.

<iframe
  src="assets/dance-box.html"
  width="800"
  height="500"
  frameborder="0"
></iframe>

### Interesting Aggregates

Grouping by genre reveals how musically distinct the six selected genres are:

| track_genre | popularity | danceability |   energy | acousticness | explicit |  valence |
| :---------- | ---------: | -----------: | -------: | -----------: | -------: | -------: |
| anime       |     48.772 |     0.537451 | 0.674108 |     0.267734 |    0.055 | 0.434463 |
| classical   |     13.055 |     0.381923 | 0.189827 |     0.920049 |        0 |  0.38105 |
| hip-hop     |     37.759 |     0.736154 |  0.68253 |     0.194175 |    0.319 | 0.551248 |
| j-dance     |     26.656 |     0.680557 | 0.703755 |     0.230469 |    0.391 | 0.564365 |
| jazz        |     13.628 |     0.509975 | 0.352954 |     0.715816 |    0.003 | 0.490289 |
| k-pop       |     56.896 |     0.647732 | 0.675654 |     0.292968 |    0.049 | 0.556915 |

This aggregation confirms the genres span a wide acoustic range: classical sits at the acoustic extreme (lowest energy, highest acousticness), k-pop carries the highest average popularity (a natural benchmark for fame-driven listening), and hip-hop and j-dance share high danceability and energy. This diversity is what makes the prediction task meaningful rather than trivial.

---

## Assessment of Missingness

### NMAR Analysis

We believe the `followers` column (and the `log_followers` / `fame_score` columns derived from it) is **NMAR (Not Missing At Random)**. Spotify's Web API officially marks the `followers` field on the Get Artist endpoint as _Deprecated_, meaning follower data is no longer guaranteed to be maintained — and artists with a genuinely low commercial presence are the least likely to have their profiles fully integrated into the API's indexing. In other words, a value is missing _precisely because the true follower count is small_, which is the defining property of NMAR: the missingness depends on the unobserved value itself, not on any other observed column.

To make this missingness **MAR** instead, we would want access to Spotify's internal **API indexing status** or **artist streaming-activity logs**. With those, we could explain the missingness using observable metadata rather than the unobserved follower count, turning the mechanism from NMAR into MAR.

### Missingness Dependency

We analyzed the missingness of `tempo` (which has non-trivial missingness) using permutation tests, choosing the test statistic based on the type of the column being tested against.

- **Depends on `track_genre`:** Using **Total Variation Distance** as the test statistic (appropriate for a categorical column) over 1,000 permutations, we observed **p = 0.0**. We reject the null hypothesis: `tempo` missingness _does_ depend on genre, consistent with a **MAR** mechanism — some genres are systematically more likely to have tempo unreported.

<iframe
  src="assets/miss-genre.html"
  width="800"
  height="500"
  frameborder="0"
></iframe>

- **Does not depend on `mode`:** Running the same permutation test against `mode` (major/minor key) gave a large p-value (≈ 0.26). We fail to reject the null hypothesis: whether a track is in a major or minor key tells us nothing about why `tempo` is missing, consistent with **MCAR** with respect to this column.

<iframe
  src="assets/miss-mode.html"
  width="800"
  height="500"
  frameborder="0"
></iframe>

The plot shows the empirical distribution of the TVD statistic under the null (genre labels shuffled), with the observed TVD marked in red.

---

## Hypothesis Testing

We tested whether explicit lyrics are associated with different popularity within the hip-hop genre.

- **Null hypothesis (H₀):** Among hip-hop tracks, explicit and non-explicit tracks have the same mean popularity; any observed difference is due to chance.
- **Alternative hypothesis (Hₐ):** Among hip-hop tracks, the mean popularity of explicit and non-explicit tracks is _different_.
- **Test statistic:** the absolute difference in mean popularity between the two groups.
- **Significance level:** α = 0.05.

The observed difference in means was **−14.687** (explicit minus non-explicit), meaning explicit hip-hop tracks were on average about 14.5 popularity points _lower_. The permutation test returned **p < 0.001**, so we **reject the null hypothesis**.

This is a counterintuitive result — one might expect explicit content to skew toward popular mainstream rap. Because this is an observational analysis and not a randomized controlled trial, we cannot conclude that explicitness _causes_ lower popularity; we can only say the data provide strong evidence that the two groups differ.

---

## Framing a Prediction Problem

We frame a **regression** problem: predict a track's **`popularity`** (a continuous 0–100 score) from its audio characteristics.

We chose `popularity` as the response variable because it is the quantity at the heart of the serendipity idea — the entire project hinges on comparing predicted popularity against actual popularity to find under-exposed tracks.

Our evaluation metric is **RMSE (Root Mean Squared Error)**. RMSE is appropriate because `popularity` is continuous and on a fixed 0–100 scale, RMSE is expressed in the same units as the target (so an error is directly interpretable as "off by N popularity points"), and squaring penalizes the large misses that matter most for our use case. We prefer it over alternatives like MAE because the serendipity logic depends on identifying the **large** gaps between prediction and reality, which RMSE emphasizes.

**Features available at the time of prediction:** all the audio characteristics (danceability, energy, loudness, etc.) are intrinsic to a track and known the moment it is uploaded, so they are valid inputs. We deliberately **exclude `fame_score`**: although they are technically available, feeding artist fame into the model would simply re-encode the filter bubble we are trying to see past. Keeping the model fame-blind is what makes its residuals meaningful as a serendipity signal.

---

## Baseline Model

Our baseline is a **Linear Regression** model implemented in a single pipeline, trained on two features:

- **`track_genre`** — _categorical_, one-hot encoded with `OneHotEncoder(drop='first')`.
- **`danceability`** — _quantitative_, passed through.

That is **1 nominal feature and 1 quantitative feature (0 ordinal)**. The only encoding required was the one-hot encoding of `track_genre`; the numeric `danceability` needs no transformation for a linear model.

We split the data with `train_test_split(test_size=0.25, random_state=42)` and evaluated generalization on the held-out test set:

| Split | RMSE   |
| ----- | ------ |
| Train | 18.485 |
| Test  | 17.864 |

We do **not** consider this model "good." A test RMSE of ~17 means predictions are off by roughly 17 popularity points on average — large on a 0–100 scale. Notably, the train RMSE is slightly _higher_ than the test RMSE, a sign that the linear model is **underfitting**: a single linear relationship cannot capture the non-linear ways audio features interact to drive popularity. This motivates a more expressive final model.

---

## Final Model

Our final model is a **Random Forest Regressor**, again wrapped in a single pipeline and trained on the **same** train/test split as the baseline (`test_size=0.25`, `random_state=42`) so the comparison is strictly like-for-like. It uses the full set of audio features plus two engineered features layered on top of the genre encoding.

**Engineered feature 1 — Speechiness binarization (`Binarizer`, threshold = 0.33).** Spotify API says that a `speechiness` below 0.33 indicates pure music while values above it begin mixing in spoken word. From a DGP standpoint, the jump from "music" to "music + speech" is a structural threshold. Therefore, a clean binary flag captures the listener-relevant distinction far better than the raw continuous value.

**Engineered feature 2 — Loudness discretization (`KBinsDiscretizer`, 4 quantile bins, ordinal).** Perceived loudness is non-linear, and the streaming industry compresses tracks into rough loudness "tiers." Binning `loudness` into four empirical quantile bands mirrors those real-world mastering tiers (from heavily compressed mainstream masters to dynamic, quiet acoustic recordings), which relates to popularity in a stepwise rather than strictly linear way.

We also impute missing `tempo` with a custom `GroupMedianImputer` that learns genre-wise medians from the training fold only (preventing leakage), reflecting that tempo ranges are tightly genre-dependent.

**Hyperparameter search** We used `GridSearchCV` with 5-fold cross-validation (scoring on negative RMSE) over:

- `max_depth` - [1, 3, 5, 7, 10, 20, None] — the primary regularization lever controlling how much each tree can memorize noise.
- `min_samples_split` - [2, 4, 6, 8, 10, 15, 20, 30] — a secondary control preventing splits on tiny anomalous subsets.

The best combination was **`max_depth = 10`** and **`min_samples_split = 2`**, selected by best cross-validated RMSE.

| Split      | Baseline (Linear) | Final (Random Forest) |
| ---------- | ----------------- | --------------------- |
| Train RMSE | 18.485            | 12.373                |
| Test RMSE  | 17.864            | 16.839                |

The final model lowers test RMSE from **17.864** to **16.839** — an improvement of about **0.76 points (~4.3%)**. The gain is modest by design: popularity is intrinsically hard to predict from audio alone because real-world popularity is driven heavily by fame, marketing, and virality — signals we intentionally excluded. However, this obstacle is the engine of the serendipity logic, not a weakness. The remaining train/test gap reflects the mild overfitting Random Forests are prone to, reined in by the `max_depth = 10` ceiling chosen via cross-validation.

---

## Fairness Analysis

We asked whether the final model predicts popularity equally well for commercially **mainstream** genres versus **niche** genres. This matters directly for the serendipity goal: if the model is systematically less accurate for niche genres, its residual-based recommendations would be least reliable exactly where hidden gems are most likely to hide.

- **Group X — Mainstream:** genres whose median popularity is at or above the overall median (**anime, hip-hop, k-pop**).
- **Group Y — Niche:** the remaining genres (**classical, jazz, j-dance**).
- **Evaluation metric:** RMSE within each group.
- **Test statistic:** the absolute difference in group RMSE, |RMSE(Niche) − RMSE(Mainstream)|.
- **Null hypothesis (H₀):** the model is fair — its RMSE is the same for both groups, and any observed difference is due to chance.
- **Alternative hypothesis (Hₐ):** the model is unfair — its RMSE differs between the two groups.
- **Significance level:** α = 0.05.

Using only the final fitted model, the test RMSE was **17.34** for niche genres and **16.38** for mainstream genres — an observed gap of **0.963**. A permutation test that shuffled the group labels 5,000 times returned **p = 0.3438**.

Since p = 0.3438 > 0.05, we **fail to reject** the null hypothesis. We do not have statistically significant evidence that the model's accuracy differs between mainstream and niche genres; a gap of this size is consistent with random chance. This is reassuring for the serendipity use case — the residual-based recommendations appear roughly as trustworthy across both segments. That said, "fail to reject" is not proof of perfect parity: a larger test set could reveal smaller real differences than this one had the power to detect.

## Serendipity Showcase — Top 3 Undiscovered Bangers per Genre

The true goal of this project is not the RMSE — it is the Serendipity Score:

$$\text{Serendipity Score} = (\hat{y} - y) \times (1 - \text{fame\_score})$$

Tracks with a high score sound like hits, come from unknown artists, and are currently flying under the radar. These are the undiscovered bangers.

---

The table below shows the top 3 tracks per genre ranked by Serendipity Score. Each entry combines a large positive residual (the model expected a hit) with a low fame score (the artist is largely unknown), making them the strongest candidates for a serendipity-driven recommendation system.

| Genre     | Track                                                       | Artist              | Serendipity | Actual | Predicted | Fame |
| --------- | ----------------------------------------------------------- | ------------------- | ----------: | -----: | --------: | ---: |
| anime     | Naruto Main Theme                                           | Minijau             |       10.54 |     33 |     48.08 | 0.30 |
| anime     | Disorder                                                    | 神崎エルザ          |       10.44 |     33 |     52.22 | 0.46 |
| anime     | 呪術廻戦                                                    | Hiroaki Tsutsumi    |        9.89 |     33 |     48.31 | 0.35 |
| classical | 3 Gymnopédies: No. 1 Lent et douloureux                     | Erik Satie          |       13.11 |      0 |     41.94 | 0.69 |
| classical | Jezus Malusieńki (Reworked by Hania Rani & Dobrawa Czocher) | Hania Rani          |       11.31 |      0 |     24.68 | 0.54 |
| classical | Under a Glass Moon                                          | Dream Theater       |       10.74 |      0 |     40.47 | 0.73 |
| hip-hop   | Rock Your Body                                              | Burna Boy           |       13.93 |      0 |     50.00 | 0.72 |
| hip-hop   | Dua Lipa                                                    | Jack Harlow         |       10.42 |      2 |     38.36 | 0.71 |
| hip-hop   | Coldplay                                                    | Lizzo               |       10.12 |      3 |     55.81 | 0.81 |
| j-dance   | Raheem Sterling                                             | Teebone             |       24.53 |      0 |     34.59 | 0.29 |
| j-dance   | Many Many Gyal                                              | Harry Toddler       |       21.13 |      0 |     35.42 | 0.40 |
| j-dance   | Rockefeller                                                 | Jahvillani          |       14.21 |      0 |     28.26 | 0.50 |
| jazz      | Love is in the Air                                          | 48th St. Collective |       24.58 |      0 |     41.25 | 0.40 |
| jazz      | Love Letter - Original Mix                                  | Living Room         |       22.28 |      0 |     38.20 | 0.42 |
| jazz      | Ganja Tribe - Original Mix                                  | Living Room         |       19.70 |      0 |     33.77 | 0.42 |
| k-pop     | Tere Pyar Ko Salam O Sanam - Jhankar Beats                  | Alka Yagnik         |       10.92 |      0 |     52.24 | 0.79 |
| k-pop     | Yaarum Illa                                                 | Yuvan Shankar Raja  |        7.08 |     34 |     65.92 | 0.78 |
| k-pop     | Sakka Podu                                                  | KK                  |        5.49 |     42 |     67.82 | 0.79 |

_Serendipity Score = (predicted − actual popularity) × (1 − fame score). Higher = more undiscovered._
