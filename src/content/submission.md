# FoodCourt+ Recommendation System


# Section 1 — Data Strategy

## 1.1 The core problem with our data

The honest starting point is that we have no ratings. Nobody was ever asked what they liked. What we do have is 18 months of purchase records and the assumption we're making is that if someone kept ordering the same thing, they probably liked it.

That assumption is fine as a starting point, but it breaks down fast. Take a user who orders Masala Dosa every morning and Chaat twice a month. Raw counts say they like dosa 5× more. But that's probably wrong the dosa is breakfast, it's light, it's cheap, it's just what you get before work. The chaat is a choice. The signal from those two chaat orders might actually be stronger than the signal from ten dosas.

This is the core problem with purchase data as a preference signal it can't tell the difference between habit and genuine preference. Everything we build in this section is an **attempt** to work around that.

---

## 1.2 Signals we extract and why

Five signals are extractable from the available schema. Each captures something different, and each introduces its own bias which we will name explicitly.

### Signal 1 : Dampened purchase frequency

The most direct signal: how many times has user *u* ordered item *i* ?

We cannot use raw counts directly. The 10th order of the same item tells us almost nothing new, we already knew the user liked it after the 2nd or 3rd. Instead we apply square-root dampening:

$$
\mathrm{preference\_score}(u,i)
=
\sqrt{\mathrm{order\_count}(u,i)}
$$

This compresses the range so that repeat orders of the same staple item don't drown out the signal from items ordered less frequently. The first order of a new item is weighted relatively more than the 10th order of a familiar one.

**Biases introduced:**

*Popularity bias* :

 *Popularity bias* — 
 
 S06 has the largest catalog at 40 items, the highest item churn at 6–8 new items a month, and two kiosk terminals while every other stall has one. More items means more chances to match any given user. More terminals means shorter queues and fewer abandoned orders,  some of those S06 orders exist simply because the wait was shorter, not because the preference was stronger. Dampening compresses the volume gap but can't separate genuine preference from structural advantage.

*Availability bias* :

 Roughly 15% of the catalog changes every month, and S06 rotates 6–8 items weekly. If a user's favourite item gets pulled from the menu, their score for that item freezes and starts decaying. The model ends up holding onto a strong preference for something that no longer exists, while the replacement item — which the user might like just as much — starts from zero with no history. 

*Price anchoring bias* :

Cheaper items naturally get ordered more often. A ₹25 vada ordered three times and a ₹180 biryani ordered once will have different dampened scores, but the biryani order probably represents a stronger deliberate choice. We don't correct for this correcting it would require knowing each user's price sensitivity, which we don't have. It's a known distortion we have to live with.

*Substitution bias* :

 When a preferred item isn't available, users order their second choice. That substitute accumulates count. Over time the model quietly learns the backup preference instead of the real one. There's no way to fix this without knowing which orders were genuine first choices and which were forced data that was never captured.

---

### Signal 2 — Time-decayed recency

A purchase from 14 months ago is less informative about current preference than 
one from last week. We weight each order by an exponential decay factor:

$$
w_o = e^{-\lambda \cdot \Delta t_o}
$$

where $\Delta t_o$ is the number of days between order $o$ and today, and 
$\lambda$ is a decay rate we set as:

$$
\lambda = \frac{\ln(2)}{180}
$$

With this decay rate, an order placed roughly six months ago contributes only 
half as much as a purchase made today.

Combining this with the dampening from Signal 1, the final preference score 
for user $u$ on item $i$ is:

$$
\mathrm{score}(u, i) = \sqrt{|O_{u,i}|} \cdot \frac{1}{|O_{u,i}|} 
\sum_{o \in O_{u,i}} e^{-\lambda \cdot \Delta t_o}
$$

where $O_{u,i}$ is the set of all orders placed by user $u$ for item $i$. 
The square root dampens the effect of order count, and the average decay 
weight reflects how recent those orders are without letting a high order 
count amplify the recency score.

**Bias introduced:**

*Recency collapse for infrequent users* :

Time decay hits infrequent 
visitors disproportionately hard. A Daily Regular visiting 5 days a week 
accumulates hundreds of decayed but still meaningful orders. An Occasional member visiting once or twice a month might make 25 orders over 18 months, 
but most of them are old after decay is applied, their effective signal 
collapses toward near-zero. The model then falls back to popular items, 
giving them the same recommendations as a walk-in stranger.

The important distinction here is that this isn't because we genuinely 
know less about their taste. An occasional user who orders the exact same 
two items from S07 every single visit has very clear preferences. The 
scoring function just mathematically erases their history because it's 
spread thin across time. That's the bias sparse visit patterns get penalised regardless of preference clarity.

We handle this by using a slower decay rate for Occasional and Dormant 
segments:

$$
\lambda = \frac{\ln 2}{360}
$$

Half life of 360 days instead of 180, preserving more of their limited 
history. The tradeoff is that stale preferences stick around longer for 
these users but for someone visiting once a month, preferences probably 
aren't changing fast enough for that to matter.

*The half life of 180 days for standard users is a starting assumption, 
not a ground truth. Post launch, we validate it by checking whether users 
whose top recommendations are dominated by orders from 6–12 months ago 
convert at lower rates than users with fresher histories. If they do, 
$\lambda$ should increase. If conversion rates are similar across history 
ages, it should decrease. This is a parameter we expect to tune after 
seeing real recommendation outcomes.*


---

### Signal 3 — Time-of-day context

S02 does 60% of its orders before 11am. The 12–2pm window alone is 35% 
of all daily orders. These aren't coincidences people eat differently 
at different times of day and the data reflects that. The person grabbing 
an idli at 8am before work is just not the same person as the one sitting 
down for a proper lunch at 1pm, even if their overall order history looks 
similar on paper.

We bucket orders into four time windows:
<center>

| Window | Hours |
|:------:|:-----:|
| Breakfast | 08:00–11:00 |
| Lunch | 11:00–15:00 |
| Evening snack | 15:00–18:00 |
| Dinner | 18:00–21:00 |

</center>


Each user's stall affinity and item preference profile is computed separately for each time window. A recommendation made at 8:30am should not surface the same items as one made at 1:15pm, even for the same user.

**Biases introduced:**

*Narrow profile for single-visit-time users* :

A user who exclusively visits during lunch will have a full lunch profile and empty profiles 
for every other window. For time windows with no personal history we 
fall back to the time-aware popularity baseline.

*Weekday vs weekend blindness* :

The four time windows don't distinguish 
between weekday and weekend visits. Weekend lunch at a mall cafeteria is 
a different context from weekday lunch more leisure time, different 
crowd, different ordering behaviour. A user whose weekend lunch orders 
look different from their weekday lunch orders will have those patterns 
averaged together into one profile. We don't split by day type in the 
initial model the data may not be dense enough per user to support 
eight windows instead of four but it's worth validating post-launch 
whether day-type explains meaningful variance.

*Occupational clustering* :

The cafeteria serves three distinct user 
types: office workers, mall staff, and shoppers. These groups naturally 
concentrate in different time windows. Office workers dominate breakfast 
and weekday lunch. Shoppers skew weekend and evening. Since we don't 
have user type in our data, time window quietly becomes a proxy for it. 
Recommendations within a window get shaped by whichever group dominates 
that window not necessarily by the individual user's actual preference.

---

### Signal 4 — Basket co-occurrence

Each transaction contains `item_ids[]`  an average of 2.3 items per 
session. The dataset already tells us something interesting: S08 (Desserts) 
has high co-occurrence with S02 and S06. People grab a dessert after their 
South Indian breakfast or their chaat. This isn't personal preference  
it's a near universal pairing behaviour that shows up regardless of who 
the user is.

We compute a co occurrence count for every item pair across all 
transactions:

$$
\mathrm{co}(i,\, j) = \bigl|\{\, t \in T : i \in t \text{ and } j \in t \,\}\bigr|
$$

where $T$ is the set of all transactions. This gives us a symmetric 
matrix of how often any two items appear in the same basket. We use 
this to surface complementary items alongside main recommendations 
a "pairs well with" layer that requires no user level data at all, 
making it particularly valuable for walk-ins with no order history.

**Biases introduced:**

*Universal pairing vs personal taste* :

Two items appearing together 
often might mean most people genuinely like both, or it might mean one 
is always the default add-on regardless of actual preference. We use 
co-occurrence only as a secondary signal it informs bundling 
suggestions, not primary rankings.

*High volume stall dominance* :

S06 has the largest catalog and highest 
order volume. Its items appear in more transactions than items from 
lower footfall stalls like S10. The co-occurrence matrix will naturally 
over represent S06 item pairs, not because those pairings are more 
meaningful but simply because S06 appears in more baskets. We normalise 
co-occurrence scores by each item's individual order frequency to 
partially correct for this.

*Session size confound* :

 A user ordering 4 items in one session 
generates 6 item pairs in the matrix. A user ordering 2 items generates 1. Users who consistently order larger baskets have disproportionate 
influence on the co-occurrence matrix not because their pairings are 
more meaningful, but because larger baskets produce more pairs 
mathematically.

---

### Signal 5 : Stall affinity

Even without item-level granularity, a user who has visited S02 eight 
times and never visited S10 has a clear cuisine level preference signal. 
I compute stall affinity as a dampened, time-decayed visit frequency 
per stall, applying both the square-root dampening from Signal 1 and 
the exponential decay from Signal 2, with the same $\lambda$ values 
used for each user segment.

This acts as a fallback when item-level data is sparse. It is most 
useful for new items entering a stall the user already has affinity 
for, and for users whose item-level history is too thin to produce 
reliable item-level scores.

**Biases introduced:**

*Floor bias* :

Upper floor stalls (S07–S10) show structurally lower 
footfall because the only access is a staircase. Upper floor stalls 
will naturally accumulate lower affinity scores across the user base 
not because users dislike them, but because the building creates mild 
friction that the model cannot see. What I am actually measuring here 
is a mix of genuine preference and physical convenience, and I have 
no way to separate the two from transaction data alone. I address 
this directly in Section 3.

*New stall bias* :

S10 opened 4 months ago. It has at most 4 months 
of visit history while every other stall has 18. A user who has 
visited S10 every week since it opened will still have a lower raw 
affinity score for it than for a ground floor stall they visited 
occasionally over 18 months. The model underranks S10 not because 
users prefer it less but because it simply has not had enough time 
to accumulate history. I address this in Section 3 alongside the 
cold start problem.

*Single stall visit pattern* :

 Users who always eat at the same stall 
produce a near one-hot affinity vector. I know they like that stall 
but have no information about anything else. For these users stall 
affinity as a fallback signal is useless for surfacing anything new. 
It just keeps confirming what I already know and contributes nothing 
toward discovery.

*Cuisine conflation* :

 Stall affinity and cuisine affinity are 
effectively the same thing in this cafeteria since each stall serves 
one cuisine. This becomes a problem at the edges. High affinity for 
S01 (North Indian) does not necessarily imply high affinity for S06 
(Street Food/Chaat) even though both skew heavily Indian. They are 
completely different eating occasions. Stall-level signals are too 
coarse to capture that distinction and I have no finer granularity 
available in the current schema.

---

## 1.3 Additional data sources worth addressing

### Loyalty points redemption log

The points log records every order that earned points, which makes it 
effectively a duplicate of the order history for standard preference 
modelling. Using it as a separate recommendation signal adds nothing new.

However the 34% redemption rate is worth treating differently. When a 
user redeems points, some degree of price friction is reduced. In theory 
what they choose to spend those points on should be a cleaner preference 
signal, closer to a genuine deliberate choice rather than a habitual one.

There are two problems with acting on this directly. First, we don't 
know the redemption threshold or the discount value, so we cannot 
confirm how much price friction is actually removed. A user redeeming 
points on their usual daily order isn't making a treat choice, they're 
just getting their habit cheaper. Second, any weight multiplier we assign 
to redemption orders right now would be a guess. We have no post-launch 
data to validate whether redemption orders actually predict preference 
better than regular orders.

Our approach is to flag redemption orders separately in the training 
data and treat the weight multiplier as a tunable parameter, starting 
at $1.5\times$ as an initial assumption. Post launch we validate this 
by checking whether items a user redeemed points on appear in their 
subsequent orders at a higher rate than items recommended through 
standard CF. If they do, the multiplier is justified and can be 
increased. If redemption orders predict no better than regular ones, 
we drop the multiplier entirely.

### Push notification open rate

The 12% open rate is not a training signal, it is an evaluation metric. We do not include notification engagement as a feature in the model. Instead, we treat it as a proxy for recommendation relevance: if our system improves, and we use push notifications to surface personalised recommendations, open rates should increase. If they don't, either the recommendations are still poor or notification fatigue has set in regardless of content quality. 

What is a strong signal is the open → order sequence: a user who opens a notification about a specific item and then orders it within the same session has demonstrated deliberate, prompted preference. We log this separately and use it as a high-confidence interaction in retraining cycles.

---

## 1.4 The metadata problem

Veg/non-veg tagging covers 70% of the catalog. Allergen tagging covers 
45%. Both are maintained by stall owners with no enforcement mechanism.

This is an operational problem, not a modelling one. I do not attempt 
to impute missing tags through ML incorrectly predicting an allergen 
tag in a food context is not an acceptable risk.

Our approach:

Items with missing veg/non-veg tags surface in recommendations but are 
flagged in the UI as "dietary info unavailable." The user sees the item 
and makes their own decision.

Items with missing allergen tags are conservatively treated as 
potentially containing common allergens and are not shown to users who 
have indicated sensitivities. This is future behaviour since no 
dietary preferences are currently collected at signup, no user has 
indicated any sensitivity yet. The handling is defined now so the 
system is ready when preference collection is added.

Even where allergen tags exist, they are self reported by stall owners 
with no verification mechanism. We surface them to users but cannot 
guarantee their accuracy. This is an operational risk that no amount 
of modelling can fix.

Tag completeness should be enforced at the point of menu submission. 
This is a form validation rule, not an ML problem. Without it dietary 
filtering will never be reliable regardless of how good the model gets.

The more significant gap is at signup. Loyalty registration captures 
name, phone, and email only. Adding a single optional veg/non-veg 
preference field would immediately improve recommendation relevance 
for every loyalty member at zero modelling cost. It is the 
highest leverage data improvement available to FoodCourt+ and it 
requires no ML work whatsoever.

---

## 1.5 Walk-in users :

Daily footfall is approximately 2,000. Daily loyalty members present 
are approximately 600. That means roughly 1,400 users per day that is 
70% of daily traffic have no loyalty account and no order history.

Collaborative filtering requires a user ID with purchase history. 
It produces nothing for walk ins.

I handle this differently depending on context:

**Kiosk context** :

 Walk-ins interact primarily through kiosks, which 
are physically located at specific stalls. A user at the S06 kiosk has 
implicitly chosen a stall or at minimum is already standing there. 
This assumption is not always clean since S06 has two terminals and 
some users may simply be at the nearest available one rather than 
specifically wanting chaat. It is however the best inference available 
without any user data. The recommendation problem narrows to what 
might this person want to add to their current order. I use basket 
co-occurrence to surface popular add-ons at that stall with no 
personal history needed.

**Time-aware popularity baseline** : 

The app may allow guest browsing 
without a loyalty account. If it does, walk-ins on the app who have 
not navigated to a specific stall get shown the top items ordered in 
the current time window over the last 14 days, ranked by order volume. 
14 days is chosen deliberately since 7 days is too noisy for lower footfall 
stalls, and 30 days risks surfacing items that have since been rotated 
off the menu given S06's weekly churn. This is not personalisation. 
It surfaces what people are actually ordering right now at this time 
of day in this cafeteria, which is meaningfully better than a static 
menu. If the app requires login, this flow only applies to loyalty 
members and this paragraph becomes irrelevant for walk-ins entirely.

**On upper floor recommendations** : 

Upper floor stalls (S07–S10) are 
included in the popularity baseline. The staircase is a single visible 
flight near the entrance and is not a significant barrier. The evidence 
is in the data itself S08 has high co-occurrence with ground floor 
stalls, which means people are already making the trip upstairs 
regularly. Excluding upper floor stalls from walk-in recommendations 
would unfairly penalise four legitimate stalls. Instead I apply a 
modest ranking adjustment described in Section 3.

---

## 1.6 Signals we chose not to use

**Mall weather API** :

 18 months of historical data is available. We deliberately exclude it from the initial model. The catalog is 273 items in a subsidised cafeteria there is no strong prior reason to believe weather drives meaningful variance in orders when controlling for day of week and time of day effects, which are already captured. Weekend footfall is 1.8× weekday regardless of weather, suggesting temporal patterns dominate. Weather data adds a feature with uncertain value and certain complexity cost. We note it as a candidate for a post-launch A/B test if the model shows unexplained variance that correlates with weather conditions.

**Cinema showtime API** :

 The mall has an attached cinema. In principle 
an upcoming showtime might push demand for quick meals beforehand, and 
cinema-goers represent a potential source of new walk-in users who 
might not otherwise visit the cafeteria. In practice we have no way 
to identify which cafeteria users are also cinema-goers from the 
current data — any correlation between showtimes and order spikes 
cannot be cleanly separated from regular evening traffic. Building 
a recommendation feature on an unverified correlation is not 
justified at launch. The more viable use case is time-targeted 
promotions coordinated with showtime schedules — but that is a 
marketing decision, not a recommendation system one. We flag it 
for FoodCourt+ to explore separately.

---


# Section 2 — Model Design

## 2.1 Algorithm choice

The system needs to answer one question: given what I know about this user right now, which items should I surface?

Before picking an algorithm, the constraints narrow the space considerably:

- 4,500 loyalty members, 273 items , this is a small matrix
- No GPU available
- 2 on-prem application servers
- No explicit ratings, only purchase history
- 15% catalog churn per month

The last point rules out deep learning approaches entirely they require GPU hardware and dense interaction data to train reliably. A neural collaborative filtering model trained on 4,500 users and 273 items with implicit feedback will overfit badly and produce worse results than a well tuned simple model.

The right algorithm for this setting is **implicit Alternating Least Squares (iALS)** a matrix factorisation method designed specifically for implicit feedback datasets.

Here is why iALS fits this problem and not a simpler or more complex approach:

**Why not simple popularity ranking?** It produces identical recommendations for every user. A Daily Regular who has ordered from S02 forty times and never touched S04 should not be shown the same list as a new signup. Popularity ranking is the fallback, not the primary model.

**Why not user-based k-NN collaborative filtering?** With 4,500 users it is computationally feasible, but k-NN requires computing similarity between users at query time. At peak load 19% of daily orders between 12:00 and 13:00, roughly 380 loyalty member sessions in one hour computing nearest neighbours on the fly is slow and unnecessary. iALS precomputes everything offline.

**Why iALS specifically?** iALS factorises the user-item interaction matrix into two lower dimensional matrices which are a user embedding matrix and an item embedding matrix. Each user and item is represented as a vector of $k$ latent factors. The dot product of a user vector and an item vector gives the predicted preference score. Training happens entirely offline. At serving time, a recommendation is just a dot product microseconds per user.

iALS also handles implicit feedback natively. It was designed for exactly this problem purchase data with no explicit ratings. It treats observed interactions as positive signal and unobserved interactions as weighted negative signal, which is the correct framing for our data.

## 2.2 Model dimensions

We set the number of latent factors $k = 50$.

The rule of thumb for matrix factorisation gives a minimum viable $k$:

$$k_{\min} \approx \sqrt{\min(|U|,\ |I|)} = \sqrt{\min(4500,\ 273)} = \sqrt{273} \approx 17$$

This is a floor, not a target — it tells us the model needs at least 17 
factors to avoid underfitting, not that 17 is optimal. In practice, iALS 
implementations use standard values of 32, 50, 64, or 128. We need enough 
factors to simultaneously capture cuisine affinity, price sensitivity, 
time-of-day behaviour, and stall loyalty at minimum four interacting 
dimensions. 32 sits too close to the floor. 128 would give the model more 
parameters than our 273-item catalog and 4,500-user base can reliably 
justify overfitting risk increases beyond that point for a dataset this 
size. 50 is the conservative middle ground: above the floor, below the 
overfitting threshold, and a standard value with established precedent in 
implicit feedback systems.

The user embedding matrix is $4500 \times 50 = 225{,}000$ floats. The 
item embedding matrix is $273 \times 50 = 13{,}650$ floats. At 4 bytes 
per float, total model size is approximately 955 KB this comfortably fits 
in RAM.

## 2.3 How many items to recommend

We surface **4 items** per recommendation request on both kiosk and app.

Average kiosk session is 245 seconds. Subtracting approximately 60 
seconds for payment processing and 30 seconds for order confirmation, 
the active browsing window is roughly 155 seconds. The recommendation 
is surfaced at the start of the stall view — the user sees it before 
they begin browsing the menu.

The constraint on how many items to show is not time — 155 seconds is 
enough to evaluate more than 4 items. The binding constraint is kiosk 
UI layout. 55% of orders come from physical kiosk terminals which 
require large, frictionless touch targets. 4 items renders as a clean 
2×2 grid — symmetrical, fully visible without scrolling, and large 
enough for accurate touch interaction. 5 items forces either an 
asymmetric 3+2 layout or a horizontal carousel that requires swiping 
to see the last item, both of which introduce friction and increase 
time-at-terminal.

On the app, 4 items renders as a clean 2×2 card grid in the 
"recommended for you" row at the top of the stall view. We keep the 
count consistent across channels to simplify the serving layer — one 
recommendation request, one response format, 4 items always.

4 slots also divide cleanly into our diversity injection strategy: 
3 confirmatory recommendations and 1 discovery item for Daily Regular 
and Frequent segments.

## 2.4 The fallback chain

iALS requires a user ID with sufficient interaction history. It produces 
unreliable embeddings for users with very few orders and nothing at all 
for users with no account. We handle this with a three-layer fallback chain:

$$
\boxed{\textbf{Layer 1} \text{ — iALS personalised recommendations}}
$$
$$
\text{Loyalty members with} \geq 3 \text{ orders}
$$
$$
\downarrow \text{ insufficient history}
$$
$$
\boxed{\textbf{Layer 2} \text{ — Stall affinity + content-based fallback}}
$$
$$
\text{Loyalty members with } 1\text{–}2 \text{ orders, or new items in a known stall}
$$
$$
\downarrow \text{ no history at all}
$$
$$
\boxed{\textbf{Layer 3} \text{ — Time-aware popularity baseline}}
$$
$$
\text{Walk-ins and new signups with zero orders}
$$

**Layer 1 :  iALS**

Standard operation for the majority of loyalty members. The serving 
layer takes the user's embedding vector, computes dot products against 
all 273 item embeddings, applies the rule-based override layer described 
in Section 3, and returns the top 4 items. The threshold of 3 orders is 
chosen because 3 orders spread across an average of 2.3 items per session 
gives the model at minimum one data point per time window enough to 
place the user in latent space with low but usable confidence.

**Layer 2 : Stall affinity and content-based fallback**

For users with 1–2 orders, iALS embeddings are unreliable there is 
not enough signal to place the user accurately in latent space. Instead 
we use their stall visit history to identify cuisine affinity and surface 
the top items from those stalls ranked by recent popularity within the 
stall. The user receives cuisine-level personalisation instead of 
item-level personalisation. It is a worse experience than Layer 1 but 
a better one than pure popularity.

For new items with no interaction history, the stall they belong to 
serves as the content signal. If user $u$ has strong affinity for S02 
and a new item is added to S02, we surface it to $u$ regardless of the 
item's interaction count. This is the primary cold start mitigation for 
new items and is explained in detail in Section 2.5.

**Layer 3 : Time-aware popularity baseline**

For walk-ins with no loyalty account and new signups with zero orders. 
Top items ordered in the current time window over the last 14 days, 
ranked by order volume. Not personalised. Surfaces what people are 
actually ordering right now at this time of day, which is meaningfully 
better than a static menu. Upper floor stalls are included with the 
ranking adjustment described in Section 3.

## 2.5 Cold start : new items

Around 15% of the catalog changes every month. S06 rotates 6–8 items 
weekly. A new item has zero interaction history when it enters the 
system. iALS has no embedding for it and will not surface it in Layer 
1 recommendations until the next retraining cycle.

Our retraining schedule is **nightly at 3:00 AM** (justified fully in 
Section 5). 3am is chosen because footfall is effectively zero  the 
hourly distribution shows only 2% of daily orders in the 20:00–21:00 
window and zero recorded after that meaning retraining runs with no 
serving load competition. The model is 955 KB and trains on 4,500 users 
in seconds on a standard CPU, so nightly retraining carries no 
meaningful compute cost.

A new item added at any point during the day will be picked up by 
the 3am retraining run and appear in Layer 1 recommendations by the 
following morning. The maximum cold start lag for iALS is therefore 
24 hours, not 7 days.

During that 24-hour window the item is surfaced through Layer 2 to 
users with stall affinity for that stall, and through Layer 3 once 
it accumulates 5 or more orders.

We retain the **new item boost** for this 24-hour window: any item 
added in the last 24 hours is automatically injected into Layer 2 
recommendations for users with affinity for that stall. The boost 
expires after 48 hours or after the item accumulates 50 interactions 
 whichever comes first. At S06's order volume, 50 interactions is 
reached in approximately 2–3 days for a well-received item, at which 
point the nightly retrain has already picked it up anyway.

For S06 specifically, an item added Monday morning enters iALS by 
Tuesday 3am and has a full week of personalised recommendation 
exposure before any rotation decision is made. The 7-day starvation 
problem from a weekly retraining cadence no longer exists.

Is the 24-hour lag acceptable to the business? Yes for all stalls 
including S06. We flag it as a known gap but not a dependency 
requiring special mitigation at launch.

## 2.6 Cold start : new users

New loyalty signups run at approximately 180 per month, roughly 6 per 
day. Each new member starts with zero interaction history.

We do not attempt to collect preferences at signup. The current flow 
captures name, phone, and email only. Adding a veg/non-veg preference 
field is an operational recommendation made in Section 1 — it is not 
a dependency of this model.

New members enter the fallback chain automatically based on their 
accumulated order count. At zero orders they receive Layer 3 
time-aware popularity recommendations. After their first or second 
order they move to Layer 2, where stall affinity from those early 
visits drives cuisine-level personalisation. Once they cross 3 orders they become eligible for Layer 1 — 
iALS picks them up at the next nightly retraining cycle, meaning 
the transition happens within 24 hours of reaching the threshold.

For Daily Regulars and Frequent members this transition happens within 
the first 1–2 weeks of joining. For Occasional members visiting 1–3 
times per month, reaching 3 orders takes anywhere from 1 to 3 months 
at the observed visit frequency — meaning they may stay on Layer 2 for 
up to 3 months before iALS picks them up. This is acceptable. Layer 2 
delivers cuisine-level personalisation from the very first order, which 
is a meaningful experience even if it is not fully personalised at the 
item level. The goal is not to rush users into Layer 1 — it is to give 
them the best experience their interaction history supports at any 
given point.

## 2.7 How the five segments are handled differently

Each segment has a different interaction density and therefore a 
different relationship with the model.

**Daily Regulars (10% of base, 5–6 visits/week)**

These users have the richest interaction history  5–6 visits per week 
over 18 months is upward of 450 sessions. iALS embeddings are stable 
and accurate for them. The risk is over-personalisation: the model 
learns their habits so precisely it stops surfacing anything new, and 
the recommendations become a mirror of what they already order rather 
than a discovery tool.

We apply a **diversity injection**: 1 of the 4 recommendation 
slots is always an item the user has never ordered, selected from stalls they 
have demonstrated affinity for. 1 out of 4 is the minimum me enough to drive discovery on every session without 
sacrificing the 4 confirmatory slots that make the recommendations feel 
relevant and trustworthy. Increasing to 2 discovery slots risks the 
recommendations feeling random, which causes users to ignore them 
entirely.

**Frequent (25%, 2–3 visits/week)**

2–3 visits per week over 18 months produces 156–234 sessions 
substantial history. iALS handles them well and embeddings are reliable. 
We apply diversity injection at the same rate as Daily Regulars  1 
discovery slot out of 4 since their visit frequency is high enough 
that without it, habit reinforcement becomes a real risk over time. 
The difference from Daily Regulars is that their profiles update 
slightly more slowly, which is acceptable given the nightly retraining 
cadence.

**Weekly (30%, ~1 visit/week)**

Approximately 72–78 sessions over 18 months. Enough history for iALS 
to produce reliable embeddings. Their visit pattern is regular enough 
that time-of-day profiles are well-defined a user who visits every 
Saturday lunch will have a strong weekend lunch profile. Time decay is 
applied at the standard rate $\lambda = \frac{\ln 2}{180}$. We do not 
apply diversity injection for this segment  at one visit per week, 
each recommendation slot is more valuable as a confirmatory signal. 
Discovery is lower priority when the user visits infrequently.

**Occasional (25%, 1–3 visits/month)**

Thin history with high recency collapse risk. We apply the reduced 
decay rate $\lambda = \frac{\ln 2}{360}$ as described in Section 1 to 
preserve more of their limited history.

Additionally, if an Occasional user's total order count is below 10, 
we blend Layer 1 and Layer 2 outputs 3 items from iALS and 2 from 
stall affinity. The threshold of 10 orders is derived as follows: at 
2.3 items per session, 10 orders covers approximately 23 unique item 
interactions, which is 8.4% of the 273-item catalog. Below this 
coverage the user's embedding vector is too sparse to place them 
reliably in latent space blending with stall affinity compensates 
for that instability.

**Dormant (10%, no visit in 90+ days)**

Dormant users have embeddings that exist but are heavily decayed. 
Serving iALS recommendations to a user whose last order was 4 months 
ago risks surfacing preferences that no longer reflect their current 
taste. We route returning Dormant users to Layer 3 time-aware 
popularity for their first session back. After that first return 
order their segment is reassessed: if they place further orders within 
14 days they are reclassified into the appropriate active segment and 
re-enter the standard fallback chain. If they do not return within 14 
days they remain Dormant.

Re-engagement nudges push notifications or loyalty point reminders
are a marketing decision outside the scope of the recommendation system. 
We flag Dormant users in the serving layer so FoodCourt+ can act on 
them through their existing notification infrastructure if they choose 
to.

## 2.8 Does time of recommendation matter?

Yes. This is not optional context — it is a primary signal.

S02 does 60% of its orders before 11am. The lunch window accounts for 35% of all daily orders. Recommending a full North Indian thali at 8:30am or an idli at 1:30pm are both poor recommendations even if the user genuinely likes both items.

Every recommendation request carries a timestamp. The serving layer maps it to one of the four time windows defined in Section 1 and selects from the user's time-window-specific preference profile. A user with a strong breakfast profile for S02 and a strong lunch profile for S01 receives completely different recommendations at 9am versus 1pm.

For users with no profile in the current time window, the fallback is the time-aware popularity baseline for that window — not the user's overall profile, which would be dominated by their most-visited window.

## 2.9 Failure modes

**Popularity collapse** :

 If diversity injection is misconfigured or 
disabled, iALS converges on recommending the same popular items to 
large clusters of similar users. The system works technically but 
delivers no discovery value. Detectable via catalog coverage metrics 
in Section 6. If coverage drops below the threshold defined there, 
the response is to verify diversity injection configuration and force 
a retraining cycle with corrected parameters.

**Embedding staleness** :

 With nightly retraining, a user who changes their eating habits 
significantly during the day receives stale recommendations 
until the 3am cycle. Acceptable given the retraining cadence — 
food preferences rarely shift within a single day.

**Floor bias leakage** : 

If the override layer in Section 3 is not 
applied correctly, upper floor stalls will be systematically 
underranked. The model cannot self-correct because the bias is 
structural, not a training error. We detect this by tracking upper 
floor stall recommendation rate as a dedicated metric. If it drops 
to near zero without a corresponding drop in upper floor order volume, 
the override layer has failed.

**Deleted item surfacing** :

 If an item is removed from the catalog 
but the recommendation store has not been updated yet, users can be 
shown items they cannot order. On a kiosk this directly damages trust 
— the user taps a recommendation and the item does not exist. We 
mitigate this by treating catalog deletions as an immediate cache 
invalidation event, not a batch update. Any item removal triggers 
an instant purge from the recommendation store regardless of the 
retraining schedule.

**New item starvation for S06** :

 If the new item boost is not 
implemented, items rotated weekly at S06 will never appear in 
personalised recommendations before they are retired. This is a 
silent failure — no standard metric catches it unless we specifically 
track new item exposure rate, which we do in Section 6.

**Retraining job failure** : 

If the nightly retraining job fails 
silently, embeddings go stale indefinitely while the system continues 
serving from the last successful model. Recommendation quality degrades 
gradually and may not be noticed until user engagement drops 
measurably. We mitigate this with a retraining health check: if no successful retraining run is logged within 25 hours, an 
alert fires. The 25-hour threshold allows for minor scheduling 
variance without false positives.

**Cache failure** :

 The serving layer depends on Redis. If Redis 
becomes unavailable, recommendation requests fall through to direct 
database queries. At peak load the 12:00–13:00 window handles 
approximately 19% of daily orders — roughly 380 loyalty member 
sessions in one hour. Direct database queries at this volume will 
degrade response time significantly. The fallback under cache failure 
is Layer 3 popularity recommendations served directly from a 
precomputed static table in PostgreSQL, which requires no Redis 
dependency and can serve under degraded conditions.

**Segment misclassification** :

A Daily Regular who takes a two-week 
holiday does not get reclassified as Dormant because we follow the 
task spec's definition of 90 days of inactivity before Dormant 
classification. 90 days is deliberately conservative — it avoids 
penalising users for normal absence patterns like holidays or 
semester breaks while still catching genuinely lapsed members.




---



# Section 3 — Constraints

Standard collaborative filtering treats all items as equally accessible, 
all users as equally reachable, and all channels as equivalent. None of 
these assumptions hold for FoodCourt+. This section identifies the 
properties of this specific system that break standard assumptions and 
documents exactly how the design accounts for each one.

---

## 3.1 Floor bias : upper floor stall underranking

The upper floor (S07–S10) is accessible only by a staircase near the 
entrance. There is no escalator. Footfall to the upper floor is 
noticeably lower than the ground floor.

The consequence for the model is subtle but compounding. Upper floor 
stalls accumulate fewer orders not because users dislike them but 
because the building creates mild friction. iALS learns from order 
history  if S07 has fewer orders than S01 across the user base, the 
model ranks S07 items lower for most users. Not because S07 is worse, 
but because friction suppressed the signal that would have told the 
model otherwise.

Left uncorrected this becomes a feedback loop. The model recommends 
ground floor stalls more, users visit ground floor stalls more, the 
model sees even more ground floor orders, and upper floor stalls get 
progressively buried. This is not a static bias it compounds with 
every retraining cycle.

**The fix: a floor-aware ranking adjustment.**

After iALS produces its raw ranked list, the serving layer applies a 
multiplicative boost to upper floor items before returning results:

$$
\mathrm{score\_adjusted}(u,\, i) 
= \mathrm{score\_iALS}(u,\, i) \times \beta_{\text{floor}}
$$

where $\beta_{\text{floor}} = 1.2$ for items belonging to S07–S10 and 
$\beta_{\text{floor}} = 1.0$ for all ground floor items.

The value $\beta = 1.2$ is derived as follows.

 Upper floor holds 4 of 
10 stalls  40% of the catalog footprint. The doc describes upper 
floor footfall as noticeably lower without giving an exact figure. 
Conservatively estimating upper floor receives around 30% of actual 
footfall against an expected 40%, the structural suppression factor 
is approximately $\frac{40}{30} \approx 1.33$. We set $\beta = 1.2$ 
deliberately below this ceiling a full correction would risk 
overcorrecting and surfacing upper floor items for users who genuinely 
prefer ground floor. Starting conservative and adjusting upward based 
on post-launch data is safer than starting aggressive.

Post-launch validation: we compare upper floor recommendation rate 
against upper floor order conversion rate. If users who receive upper 
floor recommendations convert at the same rate as those who receive 
ground floor recommendations, $\beta$ is correctly calibrated. If 
upper floor recommendations are consistently ignored, $\beta$ should 
decrease. If upper floor items rarely appear in recommendations 
despite the boost, the underlying iALS scores are too suppressed and 
$\beta$ needs to increase.

The boost is removed for users with fewer than 5 total orders who 
have never visited the upper floor. At 2.3 items per session, 5 
orders means the user has interacted with approximately 11 unique 
items around 4% of the 273-item catalog. Below this coverage 
threshold the user's preference vector is too sparse to draw any 
meaningful inference about floor preference specifically. We let 
their behaviour reveal their preference rather than applying a 
correction based on insufficient evidence.

---


## 3.2 Channel context — kiosk and app are different problems

55% of orders come from kiosk terminals. 45% come from the app.
These are not the same recommendation problem and treating them
identically is a design error.

#### Kiosk context

A user at a kiosk is physically standing at a specific stall. They
have already made a stall-level decision before the recommendation
appears. The recommendation surface on a kiosk is not a discovery
tool it is an add-on and within-stall upsell tool. Recommending
items from S02 to a user standing at the S06 kiosk is actively
unhelpful: they would have to leave, walk to S02, join a separate
queue, and pay at a separate counter.

Kiosk recommendations prioritise items from the current stall,
ranked by the user's item-level preference scores within that stall.
Where item-level history is thin, basket co-occurrence fills the
remaining slots  surfacing what other users typically add to orders
at this stall.

Cross-stall recommendations are only permitted on a kiosk when
validated co-occurrence data specifically justifies it. The only
current case in the data is S08 (Desserts) appearing frequently
alongside S02 and S06 orders. S08 is therefore permitted as a
cross-stall suggestion at the S02 and S06 kiosks. No other
cross-stall or cross-floor recommendations appear on kiosk terminals
 the behaviour has not been validated in the data for any other
combination.

Kiosk users may or may not be loyalty members. If a kiosk session
carries no user ID, the serving layer falls immediately to Layer 3
 within-stall popularity  with no personalisation attempted.

#### App context

The app requires login. Every app session carries a user ID, which
means Layer 1 is always attempted for app users. The fallback chain
still applies for users with insufficient history, but the serving
layer never operates blind on the app the way it does for anonymous
kiosk users.

A user on the app has not yet committed to a stall. Recommendations
appear on the home screen immediately after login, before stall
selection. The entire cafeteria is the decision space  this is a
genuine discovery and navigation tool. Recommendations surface items
across all stalls, apply full iALS personalisation, and apply the
floor bias adjustment from Section 3.1.

#### API routing

The API contract defined in Section 4 carries two fields that
determine which recommendation logic runs:

| Field | Type | Description |
|:------|:-----|:------------|
| `channel` | string | `"kiosk"` or `"app"` |
| `stall_id` | string or null | Populated for kiosk requests only |

The serving layer reads these fields first and routes to the correct
logic before any scoring begins. A kiosk request with a valid
`stall_id` routes to within-stall logic. An app request routes to
full cross-stall iALS personalisation.

---

## 3.3 Seating and queue independence

Each stall operates with its own queue, billing counter, and
dedicated seating area. Stalls do not share seating.

This has a practical implication most recommendation systems
would miss entirely. Ordering from two different stalls in one
visit means joining two queues, paying twice, and potentially
sitting in two different parts of the cafeteria. For a user on
a 30-minute lunch break this is not a minor inconvenience —
recommending items spread across three different stalls is
operationally unrealistic regardless of how well the model
thinks they match the user's preferences.

**The fix:** cross-stall recommendations are capped at a maximum
of 2 stalls per recommendation set. If the top 4 ranked items
span 3 or more stalls, the serving layer collapses the set to
the top 2 stalls by aggregate affinity score and fills all 4
slots from those stalls only.

The documented exception is S08. Desserts has high co-occurrence
with both S02 and S06 — users demonstrably make the separate
trip upstairs for dessert after their main meal. This behaviour
is already validated in the data, so S08 is permitted as a third
stall in the recommendation set when the primary stall is S02
or S06. It is not permitted as a third stall in any other
combination.

---

## 3.4 Catalog churn and stale recommendations

Around 15% of the catalog changes every month. S06 rotates 6–8 items
weekly. S08 changes on an ad hoc basis with no fixed schedule.

Two distinct failure modes arise from this churn rate and they
require different fixes.

### Deleted items appearing in recommendations

If an item is removed from the menu but still exists in the
recommendation store, a user will be shown something they cannot
order. On a kiosk this is a direct trust failure — the user taps a
recommendation and the item does not exist.

We treat catalog deletions as immediate cache invalidation events.
Any item removal triggers a purge request to Redis within the same
request lifecycle as the deletion — not deferred to the nightly
retraining cycle. In practice this means the window between a
catalog deletion and the item disappearing from recommendations is
bounded by the Redis round-trip latency, typically under 5ms.

If the Redis purge fails — for example because Redis is temporarily
unavailable — the item remains in the recommendation store despite
being deleted from the catalog. To catch this, a nightly
reconciliation job runs alongside the retraining cycle. It compares
every item in the recommendation store against the live catalog and
purges any item that no longer exists. This is the safety net, not
the primary mechanism. The primary mechanism is the immediate purge
on deletion.

### New items entering the catalog

Catalog additions also trigger an immediate event. When a new item
is registered in the catalog it is immediately written to the
system with a zero interaction count, making it eligible for Layer
2 surfacing and the new item boost described in Section 2.5 right
away. It does not wait for the nightly retraining cycle to become
visible in the system.

### Retired item embeddings polluting future training runs

After an item is deleted its embedding persists in the iALS model
until the next nightly retraining. This is not a serving problem —
the cache purge handles that — but it is a training quality issue.
Future retraining runs that include interaction history for
non-existent items produce embeddings that are partially shaped by
ghost signal.

We handle this by flagging deleted items in the training data
pipeline. Flagged items are excluded from future item embedding
computation but their historical interactions are retained for user
embedding computation only. A user who frequently ordered a
now-deleted S06 item should still have that preference reflected in
their user vector, even if the item itself no longer has an embedding.

---

## 3.5 S10 new stall bias

S10 (Japanese/Korean) opened 4 months ago. Every other stall has 18
months of interaction history. S10 has at most 4.

The problem is not just cold start at the item level — it is cold
start at the stall level. S10 has accumulated only $\frac{4}{18}
\approx 22\%$ of the interaction history that every other stall has.
In the worst case its items are suppressed to roughly $\frac{1}{4.5}$
of the signal density they would have if the stall had been open the
full 18 months. iALS will rank S10 items lower across the board not
because users dislike Japanese or Korean food but because the data
simply has not had time to reflect actual demand.

We apply a new stall boost multiplicatively alongside the floor boost:

$$
\mathrm{score\_adjusted}(u,\, i \in S10)
= \mathrm{score\_iALS}(u,\, i) \times 1.2 \times 1.15
= \mathrm{score\_iALS}(u,\, i) \times 1.38
$$

The $1.2$ factor is the floor boost from Section 3.1. The $1.15$
factor is the new stall correction. The suppression factor of 4.5
would suggest a far larger correction is theoretically justified,
but applying it in full would risk forcing S10 items onto users
who may genuinely not want Japanese or Korean food. We set
$\beta_{\text{new}} = 1.15$ as a conservative nudge that brings
S10 items into visible range without dominating recommendation
lists. Post-launch validation follows the same approach as the
floor boost — if S10 recommendations consistently go unacted on
after the boost is applied, the boost is too aggressive. If S10
items never surface despite the boost, $\beta_{\text{new}}$ needs
to increase.

The new stall boost expires when S10 reaches 12 months of
operation. At that point S10 will have accumulated
$\frac{12}{18} \approx 67\%$ of the signal parity of a full-tenure
stall — sufficient for iALS to produce reliable embeddings without
correction. This is a calendar-triggered expiry because the bias
is temporal: it exists because the stall is young and goes away
as the stall matures regardless of order volume.

One limitation worth flagging: after 12 months the boost expires
and S10 competes without correction. If S10 is still
underperforming at that point, we have no mechanism to distinguish
residual structural bias from genuine low demand. Post-launch
monitoring of S10's recommendation-to-order conversion rate
relative to other stalls is the only way to catch this.

---

## 3.6 Kiosk session latency constraint

Average kiosk session duration is 245 seconds. The recommendation
must be rendered and visible before the user begins browsing the
menu — within the first 2 seconds of the session, which is the
first cognitive moment where the user's attention is available
before it lands on the menu itself.

Working backwards: React Native UI rendering takes approximately
100ms for a component update. Network round-trip on the mall's
internal leased line adds approximately 20ms. To hit a 2-second
visibility target the API must respond within
$2000 - 100 - 20 = 1880\text{ms}$. We set the API response target
at 200ms — well within that budget — to ensure snappy UX even
under mild server load and leave headroom for any unexpected
latency spikes.

The same 200ms target applies to app recommendations. App sessions
average 320 seconds and recommendations appear on the home screen
immediately after login, giving slightly more time before the user
begins navigating — but we apply the same target for consistency
and simplicity across both channels.

With precomputed recommendations stored in Redis, serving is a
single cache lookup — sub-millisecond under normal conditions.
The 200ms budget presents no challenge in the happy path. The
constraint becomes relevant only under Redis failure, which is
handled by the PostgreSQL static fallback described in Section 2.9.

---

## 3.7 Recommendation consistency across retraining cycles

The nightly retraining run at 3:00 AM updates all user and item 
embeddings simultaneously. A user who opens the app at 2:59 AM and 
again at 3:01 AM may receive noticeably different recommendations — 
not because their behaviour changed but because the model just 
retrained.

In practice this edge case affects near-zero users. The 20:00–21:00 window carries 2% 
of daily orders and traffic after that is effectively zero. Almost 
no user will straddle a retraining boundary. We do not version-lock 
recommendations to individual sessions  the added complexity is not 
justified by the near-zero exposure. Recommendation freshness after 
retraining is more valuable than strict within-session consistency 
at 3 AM.

## 3.8 S02 time concentration bias

S02 (South Indian) does 60% of its orders before 11am. This is not
a minor skew — the overwhelming majority of S02 interaction data
sits in the breakfast time window. A standard CF model trained on
aggregate interaction history without time-window separation will
learn S02 as a globally popular stall and surface S02 items at all
times of day based on that aggregate signal.

This is the reason time-window profile decomposition, designed in
Section 1.3, is not optional for this system — it is especially
critical for S02. Without it, S02's breakfast dominance bleeds into
every time window and inflates its affinity score at hours where
most users have no intention of ordering South Indian food.
Recommending idli and dosa at 1:30pm to a user whose entire S02
history is breakfast orders is a poor recommendation even if their
aggregate S02 affinity score is high. The affinity is real — the
timing context makes it irrelevant.

The serving layer enforces time-window scoping for all stall affinity
lookups. A user's S02 affinity in the lunch window is computed only
from their lunch-window S02 interactions. If that count is low, the
affinity score is naturally low — we do not suppress it artificially,
but we do not inflate it with breakfast data either. A user who has
visited S02 twice during lunch in 18 months has a low but real lunch
affinity for S02. That is the correct signal to serve.

## 3.9 The loyalty asymmetry constraint

Daily footfall is approximately 2,000. Daily loyalty members present
are approximately 600. On any given day 1,400 users — 70% of daily
traffic — are completely invisible to the personalisation layer.

This is not just a cold start problem. Cold start implies the user
will eventually accumulate history and become visible. Under the
current signup rate of approximately 180 new members per month,
the walk-in population remains the majority of daily traffic for
the foreseeable future. Many of these users may never sign up.

The practical consequence is a hard ceiling on system impact. The
recommendation system as designed can only deliver personalised
experiences to at most 30% of daily users on any given day. The
remaining 70% receive Layer 3 popularity recommendations regardless
of how good the model gets. No amount of modelling improvement
raises this ceiling — it is determined entirely by loyalty programme
penetration.

This changes how success should be measured. Aggregate
recommendation conversion rate will be heavily diluted by the 70%
walk-in population receiving non-personalised recommendations.
Evaluation metrics in Section 6 are therefore computed separately
for loyalty members and walk-ins — blending them would make the
personalisation layer look worse than it is and the popularity
baseline look better than it is.

The only lever that raises this ceiling is loyalty sign-up rate.
A prompt encouraging sign-up at the end of a kiosk transaction —
when the user has just had a positive experience — is the simplest
available intervention. It is outside the scope of this
recommendation system but is the single highest-impact action
available to FoodCourt+ if they want to expand personalisation
reach.



---



# Section 4 — Pipeline

This section covers the full flow from raw transaction data to a rendered recommendation on the user's screen. It is divided into five parts: the offline training component, the event-driven update triggers, the serving layer, the API contract, and the fallback chain as it operates within the live system.

---

## 4.1 Offline training component

The offline component runs every night at 3:00 AM as a scheduled batch job. This window is chosen because the hourly distribution shows near-zero traffic after 21:00 — the nightly job competes with no live serving load.

The job runs in five sequential steps:

**Step 1 — Data extraction**

The job reads all loyalty member order records from PostgreSQL for
a rolling 18-month window — records from the last 18 months
relative to the current date. Older records are excluded from
training but retained in the database. This keeps the training
dataset size constant as the system matures and prevents the
nightly job from slowing down over time.

Each record contains:
~~~
user_id, timestamp, stall_id, item_ids[],
payment_method, dine_in_flag, kiosk_id
~~~
The estimated record count is derived as follows. Weighted average
visit frequency across segments is approximately 6.85 visits per
member per month. Over the average loyalty tenure of 8.4 months
across 4,500 members, the total record count is approximately:

$$
4{,}500 \times 6.85 \times 8.4 \approx 259{,}000 \text{ records}
$$

At approximately 54 bytes per record — derived from the field 
sizes computed in Section 6.1 — the extraction reads roughly 
14 MB from PostgreSQL, completing in seconds.

Deleted items are flagged in the database at the time of deletion.
The extraction query excludes flagged items from item embedding
computation but retains their interactions for user embedding
computation.

**Step 2 — Signal computation**

For every (user, item) pair in the extracted data, the pipeline
computes the preference score defined in Section 1:

$$
\mathrm{score}(u, i) = \sqrt{|O_{u,i}|} \cdot \frac{1}{|O_{u,i}|}
\sum_{o \,\in\, O_{u,i}} e^{-\lambda \cdot \Delta t_o}
$$

Scores are computed separately per time window — breakfast, lunch,
evening snack, dinner — producing four interaction matrices per
user rather than one aggregate matrix. The decay rate $\lambda$
applied depends on the user's segment: standard rate
$\frac{\ln 2}{180}$ for Daily Regulars, Frequent, and Weekly
segments; reduced rate $\frac{\ln 2}{360}$ for Occasional and
Dormant segments.

Redemption orders are identified from the loyalty points log and
assigned a $1.5\times$ weight multiplier before score computation.

**Step 3 — iALS training**

The pipeline trains four separate iALS models — one per time window
— on the computed interaction matrices. Each model uses $k = 50$
latent factors and runs on CPU.

iALS time complexity per iteration is approximately
$O(k^2 \cdot |R|)$ where $|R|$ is the number of non-zero
interactions. With $|R| \approx 259{,}000$ and $k = 50$:

$$
\frac{k^2 \times |R|}{10^9}
= \frac{2{,}500 \times 259{,}000}{10^9}
\approx 0.65 \text{ seconds per iteration}
$$

Over 15–20 iALS iterations, each model trains in approximately
10–15 seconds. All four time-window models complete in under
60 seconds total.

The output is four pairs of embedding matrices per time window:

- User embedding matrix: $4{,}500 \times 50 = 225{,}000$ floats
- Item embedding matrix: $273 \times 50 = 13{,}650$ floats

**Step 4 — Recommendation pre-computation**

Rather than computing recommendations at serving time, we
pre-compute the top 10 recommendations per user per time window
immediately after training. For each user $u$ and time window $w$:

$$
\mathrm{top10}(u, w) = \underset{i}{\mathrm{argtop10}}
\left( \mathrm{score\_adjusted}(u, i, w) \right)
$$

where $\mathrm{score\_adjusted}$ applies the floor boost, new stall
boost, and diversity injection defined in Section 3. We store 10
rather than 4 to give the serving layer flexibility to apply
real-time filters — such as removing items deleted since the last
training run — without dropping below 4 results.

**Step 5 — Cache population**

Pre-computed recommendations are written to Redis with the key
structure:
rec:{user_id}:{time_window}

Each key stores a JSON array of 10 objects — not just item IDs —
containing the fields needed by the serving layer:

```json
[
  { "item_id": "I0042", "stall_id": "S02", "score": 0.87 },
  ...
]
```

Storing stall_id alongside item_id allows the serving layer to
apply the 2-stall cap from Section 3.3 without additional database
lookups. TTL is set to 25 hours — slightly longer than the
retraining cycle. If a retraining job fails and the TTL expires,
all Layer 1 keys vanish and users fall to Layer 2 or Layer 3. This
is the condition that triggers the 25-hour health check alert
defined in Section 4.2 — the two are designed together.



**Estimated job duration**

| Step | Estimated time |
|:-----|:--------------|
| Data extraction | ~5 seconds |
| Signal computation | ~30 seconds |
| iALS training (4 models) | ~60 seconds |
| Pre-computation (18,000 user-window pairs) | ~10 seconds |
| Redis cache population (18,000 writes) | ~15 seconds |
| **Total** | **~2 minutes** |

The 3:00 AM window provides over 5 hours of buffer before peak
morning traffic begins at 08:00.

## 4.2 Event-driven update triggers

The nightly batch job handles the standard retraining cycle. Four
types of events require immediate action outside this schedule.

### Catalog deletion

Catalog mutations — additions and deletions — arrive via a catalog
management interface used by stall owners. Each mutation triggers
an event in the system immediately upon save.

When a stall owner deletes an item:

1. The item is marked as deleted in PostgreSQL with a deletion
   timestamp
2. The system looks up a reverse index maintained in Redis:
   `item_rec_index:{item_id}` which maps to all
   `rec:{user_id}:{time_window}` keys containing that item
3. The item is purged from each affected key immediately
4. If the Redis purge fails, the item is added to a reconciliation
   queue processed by the nightly job before cache population

The reverse index is populated during Step 5 of the nightly job
alongside the recommendation arrays. It allows targeted purges
without scanning all 18,000 recommendation keys on every deletion.

### Catalog addition

When a new item is added to the catalog:

1. The item is registered in PostgreSQL with a zero interaction
   count and a creation timestamp
2. The item is immediately eligible for Layer 2 surfacing via
   stall affinity — no retraining needed
3. If the item was added within the last 48 hours, the new item
   boost flag is set and the item is injected into Layer 2
   recommendations for users with affinity for that stall. The
   48-hour window matches the boost duration defined in Section 2.5

### User segment change

Segment classification is recomputed nightly as part of the batch
job — it is not event-driven. A user who stops visiting does not
get reclassified as Dormant until 90 days of inactivity have
elapsed, at which point the next nightly job updates their segment
and routes them to Layer 3 on their next visit.

### Retraining health check

If no successful retraining run is logged within 25 hours, an
alert fires. The 25-hour threshold allows for minor scheduling
variance without false positives. Under a failed retraining
scenario, the system continues serving from the last successful
cache — recommendations become stale but the system does not
go down. If the TTL on Redis keys expires before the next
successful run, users fall through to Layer 2 and Layer 3
automatically.

---

## 4.3 Serving layer

The serving layer handles live recommendation requests from the app
and kiosk. It runs on the two application servers and is the only
part of the system that operates under real-time latency constraints.

### Step 1 — Request parsing

The serving layer receives a REST request containing:
user_id, channel, stall_id (kiosk only), timestamp

The timestamp is mapped to one of the four time windows. The
channel field determines which recommendation logic path is
followed — kiosk routes to within-stall logic, app routes to
full cross-stall personalisation as defined in Section 3.2.

### Step 2 — User classification

The serving layer looks up the user's profile from Redis:
profile:{user_id} → {segment, order_count, last_visit}

If the profile key does not exist — new user or Redis failure —
the serving layer treats the user as order_count = 0 and routes
directly to Layer 3. Otherwise, order_count determines the
starting layer:

<center>

| order_count | Starting layer |
|:------------|:--------------|
| 0 or no user_id | Layer 3 |
| 1–2 | Layer 2 |
| ≥ 3 | Layer 1 attempted |

</center>

### Step 3 — Cache lookup (Layer 1)

For Layer 1 users the serving layer performs a Redis lookup:
GET rec:{user_id}:{time_window}

If the key exists and returns a valid array, the serving layer
applies real-time filters in order:

1. Remove any items flagged as deleted since the last retraining
   run
2. Apply the 2-stall cap from Section 3.3 — if the remaining
   items span more than 2 stalls, collapse to the top 2 stalls
   by aggregate score
3. Verify that at least 1 of the 4 items has never been ordered
   by the user — diversity injection was applied during
   pre-computation but deletions or the stall cap may have
   removed the discovery item. If all 4 remaining items are
   previously ordered, pull the next unordered item from the
   stored top-10 array

The filtered result is truncated to 4 items and returned.
If the key does not exist or has expired, fall through to Layer 2.

### Step 4 — Fallback handling

**Layer 2** : Look up the user's stall affinity profile from
Redis, identify the top stalls for the current time window, and
return items ranked by order volume within the current time window
over the last 14 days, filtered to items the user has not
previously ordered. For new items with the boost flag set, inject
them into the result before ranking.

**Layer 3** : Return the precomputed time-window popularity list
stored as a static table in PostgreSQL top items ordered in the
current time window over the last 14 days across all stalls. This
requires no Redis dependency and serves under fully degraded
conditions.

### Step 5 — Response assembly and return

The serving layer assembles the final 4-item array and returns it.
Responses are not cached at the HTTP level — each request hits
Redis directly. Given sub-10ms Redis lookup times, HTTP caching
would save negligible latency and risk serving stale results within
a session.

Total serving time target: under 200ms. Under normal Redis
operation, actual serving time is sub-10ms.

---

## 4.4 API contract

The recommendation API exposes a single versioned REST endpoint:
POST /api/v1/recommendations

The `/v1/` prefix versions the contract explicitly. Kiosk
terminals are physical devices that cannot be updated instantly —
versioning ensures future breaking changes can be deployed as
`/api/v2/` without disrupting terminals still running older
firmware.

POST is used rather than GET to keep user identifiers out of
URL query strings and server access logs.

### Request body

```json
{
  "user_id": "U4821",
  "channel": "kiosk",
  "stall_id": "S06",
  "timestamp": "2026-05-11T12:34:00+05:30"
}
```

| Field | Type | Required | Description |
|:------|:-----|:---------|:------------|
| `user_id` | string | No | Loyalty member ID. Absent for walk-ins |
| `channel` | string | Yes | `"kiosk"` or `"app"` |
| `stall_id` | string | No | Required if channel is `"kiosk"` |
| `timestamp` | ISO 8601 | Yes | Client timestamp for time window mapping |

All requests must carry a valid session token in the
`Authorization` header. Kiosk terminals use a device-level token
issued at terminal setup. App clients use the loyalty member
session token from login.

Clients must set a request timeout of 500ms. If no response is
received within 500ms, the client renders without a recommendation
strip rather than blocking the session. This is the client-side
complement to the 200ms server-side response target defined in
Section 3.6 — the gap accounts for network overhead and provides
a buffer for mild server load spikes.

### Response body

```json
{
  "request_id": "rec_8f3a91bc",
  "recommendations": [
    {
      "item_id": "I0042",
      "stall_id": "S02",
      "item_name": "Masala Dosa",
      "layer": 1
    }
  ],
  "time_window": "breakfast",
  "personalised": true
}
```

### Response fields

| Field | Description |
|:------|:------------|
| `request_id` | Unique identifier for this recommendation response. Used to attribute subsequent orders to the recommendation that preceded them in Section 6 evaluation |
| `recommendations` | Array of exactly 4 items ordered by adjusted score. Score is not exposed — ordering is server-side |
| `time_window` | Time window the request was mapped to |
| `personalised` | `true` if Layer 1 or Layer 2, `false` for Layer 3 |
| `layer` | Per-item field. Which fallback layer produced this item: 1, 2, or 3 |

### Error responses

| HTTP code | Condition |
|:----------|:----------|
| 400 | Missing required fields |
| 401 | Missing or invalid session token |
| 422 | Invalid channel value or missing stall_id for kiosk |
| 503 | Both Redis and PostgreSQL unavailable |

The `layer` field and `request_id` are both included deliberately.
`layer` allows the client UI to render a visual distinction between
personalised and popularity-based recommendations. `request_id`
allows the evaluation pipeline in Section 6 to attribute orders
to the specific recommendation response that preceded them —
without it, conversion tracking cannot be done reliably.

---

## 4.5 Fallback chain in the live system

The fallback chain defined in Section 2.4 operates as follows
within the live serving layer. Each step is attempted in order
and the first successful result is returned.

$$
\boxed{\textbf{Layer 1} \text{ — Redis cache hit, order\_count} \geq 3}
$$
$$
\downarrow \text{ cache miss, insufficient history, or } order\_count < 3
$$
$$
\boxed{\textbf{Layer 2} \text{ — Stall affinity from Redis profile, order\_count} \geq 1}
$$
$$
\downarrow \text{ no profile, zero orders, or Redis unavailable}
$$
$$
\boxed{\textbf{Layer 3} \text{ — Static popularity table from PostgreSQL}}
$$
$$
\downarrow \text{ PostgreSQL also unavailable}
$$
$$
\boxed{\textbf{HTTP 503 — no recommendations returned}}
$$

If Redis is unavailable entirely, both Layer 1 and Layer 2 fail
since both depend on Redis for cached recommendations and stall
affinity profiles respectively. In this scenario the serving
layer skips directly to Layer 3 — the PostgreSQL static popularity
table has no Redis dependency and serves under degraded conditions.

The 503 case requires both Redis and PostgreSQL to be
simultaneously unavailable. Assuming independence and a 99.2%
uptime SLA, the probability of simultaneous failure is:

$$
0.008 \times 0.008 = 0.000064
$$

This is 0.0064% of the time  approximately 34 minutes per year.
When it occurs, the client UI renders the standard menu without
a recommendation strip rather than showing an error to the user.

During the 3am cache repopulation, some Redis keys are updated
before others. A request arriving mid-repopulation may receive
recommendations from the new model for one user and the old model
for another. This is accepted behaviour  the 3am window carries
near-zero traffic and recommendation consistency across users
during retraining is not a design requirement.



---



# Section 5 — Evaluation

Knowing whether a recommendation system is working is harder than it sounds. A system that recommends the four most popular items to
everyone will show strong raw conversion numbers popular items get ordered frequently regardless of whether the recommendation
caused the order. Naive metrics reward this. Our evaluation
framework is designed so that a popularity baseline cannot game it.

---

## 5.1 The popularity baseline trap

Before defining metrics we establish two baselines.

**Naive baseline** — recommends the same 4 items by overall
order volume to every user, regardless of history or time
of day. This is trivially easy to beat and not a meaningful
test of personalisation.

**Strong baseline (Layer 3)** — recommends the top 4 items
by order volume within the current time window over the
last 14 days. This is time-aware but not personalised.
It is a genuinely useful default and a much harder
comparison point. Beating this baseline is what actually
demonstrates that personalisation adds value.

The strong baseline will score well on raw conversion rate
because popular items get ordered frequently regardless of
whether the recommendation caused the order. It will score
well on click-through rate because familiar items get
tapped. It will score poorly on per-user novelty because
it never shows anyone something new, on catalog coverage
because it surfaces the same items repeatedly, and on
personalisation lift by definition since it has none.

Every metric we define must be one the strong baseline
cannot game. If our system scores only marginally better
than Layer 3 on a given metric, that metric is measuring
time-aware popularity, not personalisation. We track both
baselines explicitly and report against them on every
metric.

---

## 5.2 Primary metrics

### Recommendation conversion rate (RCR)

The fraction of recommendation responses where at least one
recommended item was ordered in the same session:

$$
\mathrm{RCR} = \frac{\text{sessions where} \geq 1
\text{ recommended item was ordered}}
{\text{total sessions where recommendations were shown}}
$$

Attribution uses the `request_id` field from the API response
defined in Section 4.4. An order is attributed to a
recommendation if the `request_id` from the preceding
recommendation response is logged alongside the order within
the same session. A session is defined as a single continuous
interaction — one kiosk visit or one app open — bounded by
a 30-minute inactivity timeout. Orders placed more than 30
minutes after the recommendation response are not attributed
to it.

RCR alone is not sufficient — the strong baseline will score
well here because popular items get ordered frequently
regardless of the recommendation. RCR must always be reported
alongside personalisation lift.

### Personalisation lift

The difference in RCR between Layer 1 and Layer 3:

$$
\mathrm{Lift} = \mathrm{RCR}_{\text{Layer 1}}
- \mathrm{RCR}_{\text{Layer 3}}
$$

A positive lift means personalisation is adding value beyond
what time-aware popularity alone delivers. At launch any
positive lift is acceptable. At 6 months we expect lift to
be statistically significant — tested using a two-proportion
z-test on Layer 1 vs Layer 3 sessions. With approximately
600 daily loyalty sessions, 6 months of data provides
roughly 108,000 sessions — sufficient power for this test.

If lift is zero or negative, the personalisation layer is
not working and the model needs retuning.

### Catalog coverage

The fraction of the active catalog appearing in at least
one recommendation over a rolling 14-day window:

$$
\mathrm{Coverage} = \frac{\text{distinct items recommended
in last 14 days}}{\text{average active catalog size
over last 14 days}}
$$

The denominator uses the average active catalog size rather
than a snapshot — with 15% monthly churn, items added and
retired within the window should not count against coverage.

The strong baseline produces coverage of approximately
$\frac{4}{273} \approx 1.5\%$ — it surfaces the same 4
items to everyone. Our target is derived from the diversity
injection logic: 600 daily sessions × 14 days × 1 discovery
slot per 4 recommendations = 2,100 discovery recommendation
slots over the window. Even with significant repetition
across users, reaching 68 unique items — 25% of 273 — in
14 days is conservative. Target at 30 days is above 25%
and at 6 months above 40%.

Coverage directly detects popularity collapse — the failure
mode from Section 2.9 where diversity injection breaks and
the model converges on the same items for everyone.

### Per-user novelty rate

The fraction of recommended items the user has never
previously ordered:

$$
\mathrm{Novelty} = \frac{\text{recommended items never
ordered by this user}}
{\text{total items recommended to this user}}
$$

Targets by segment:

| Segment | Novelty target | Reasoning |
|:--------|:--------------|:----------|
| Daily Regulars | ≥ 25% | 1 discovery slot out of 4 always enforced |
| Frequent | ≥ 25% | Same diversity injection applies |
| Weekly | ≥ 10% | Confirmatory recommendations prioritised |
| Occasional | ≥ 5% | Thin history — confirmatory is more valuable |
| Dormant | Not measured | Layer 3 only on return — no personalisation |

The strong baseline scores near-zero novelty for Daily
Regulars who have already tried the top items — it keeps
recommending what they already know. This is the metric
where personalisation should show its clearest advantage
over the baseline.

---

## 5.3 Per-layer evaluation

Since the `layer` field is returned per item in every API
response, we can measure conversion rate separately for each
layer without additional instrumentation:

| Layer | What it measures | Expected RCR |
|:------|:----------------|:-------------|
| Layer 1 | iALS personalisation quality | Highest |
| Layer 2 | Stall affinity fallback quality | Middle |
| Layer 3 | Time-aware popularity baseline | Floor |

We expect Layer 1 RCR > Layer 2 RCR > Layer 3 RCR. One
important caveat: this comparison is not perfectly controlled.
Layer 1 serves users with 3 or more orders — by definition
more engaged members who visit frequently and are more likely
to order regardless of recommendations. Layer 3 serves
walk-ins and new users who may be less committed. Some of
the Layer 1 advantage will reflect user engagement, not
purely personalisation quality. Layer RCR comparisons are
diagnostic signals, not causal measurements. The 6-month
A/B holdout test defined in Section 5.6 is the only clean
causal comparison available.

With that caveat stated, the ordering still provides useful
diagnostics:

**If Layer 2 RCR > Layer 1 RCR** — iALS embeddings are not
adding value over stall affinity alone. The likely causes
are insufficient interaction history reaching Layer 1 (the
≥3 order threshold may be too low, admitting users whose
embeddings are still unreliable) or a value of $k$ that
is too high for the current data density, producing
overfitted embeddings. Response: lower the Layer 1
eligibility threshold to ≥5 orders and retrain with
$k = 32$ to compare.

**If Layer 3 RCR > Layer 2 RCR** — stall affinity profiles
are worse than just showing popular items. The likely cause
is affinity profiles that are too sparse or too stale —
users on Layer 2 have only 1–2 orders, which may not be
enough to identify meaningful stall affinity. Response:
review whether the Layer 2 affinity logic is correctly
applying time-window scoping, and consider raising the
Layer 1 threshold so fewer users fall to Layer 2.

**If all three layers show declining RCR over time** —
the system as a whole is losing relevance. This is a
signal to review catalog changes, check whether the
recommendation store contains stale or deleted items,
and verify retraining is completing successfully.

---

## 5.4 System health metrics

These metrics detect specific failure modes defined in
Section 2.9.

### Upper floor recommendation rate

The fraction of recommendation responses containing at
least one upper floor item, computed separately for users
who have previously visited the upper floor and those
who have not:

$$
\mathrm{UFRR} = \frac{\text{recommendations containing}
\geq 1 \text{ upper floor item}}
{\text{total recommendations served}}
$$

Splitting by prior upper floor visit history gives a
cleaner signal than an aggregate rate. For users who have
previously visited S07–S10, UFRR should be meaningfully
above zero — these users have demonstrated willingness to
go upstairs and the floor boost should be surfacing upper
floor items to them regularly. For users who have never
visited the upper floor, low UFRR is expected and correct.

If aggregate UFRR drops to near zero without a
corresponding drop in upper floor order volume, the floor
bias override layer has failed and needs investigation.

### New item exposure rate

The fraction of new items that have appeared in at least
one recommendation within 48 hours of addition — matching
the new item boost window defined in Section 2.5:

$$
\mathrm{NIER} = \frac{\text{new items appearing in} \geq 1
\text{ recommendation within 48 hours of addition}}
{\text{total new items added}}
$$

If NIER drops toward zero for S06 items, the new item
boost has failed and S06 rotation is invisible to the
recommendation layer. Target: above 80% within 48 hours
of item addition.

### Retraining freshness

Binary metric — did the nightly retraining job complete
successfully within the last 25 hours? Fires an alert if
not. Tracked as a timestamp check on the retraining log.
The 25-hour threshold allows for minor scheduling variance
without false positives.

### Layer distribution

The fraction of recommendation responses served by each
layer over a rolling 7-day window. This metric must be
computed separately for loyalty member sessions and all
sessions — the two populations have very different expected
distributions.

**Loyalty member sessions only:**

| Layer | Expected share |
|:------|:--------------|
| Layer 1 | ~70–75% |
| Layer 2 | ~15–20% |
| Layer 3 | ~5–10% |

**All sessions (including walk-in kiosk):**

| Layer | Expected share |
|:------|:--------------|
| Layer 1 | ~20–25% |
| Layer 2 | ~5–10% |
| Layer 3 | ~65–70% |

Walk-ins constitute roughly 1,400 of 2,000 daily sessions
— 70% of traffic — and all receive Layer 3. Blending
loyalty and walk-in sessions into one distribution masks
this and makes Layer 3 dominance look like a failure when
it is expected behaviour.

Alert threshold: if Layer 3 share among loyalty member
sessions exceeds 40% over a 1-hour rolling window, fire
an alert. This indicates either a Redis failure causing
mass cache misses or a retraining job failure that expired
all Layer 1 keys.

---

## 5.5 What success looks like at 30 days

At 30 days post-launch the system has completed approximately
30 nightly retraining cycles. It is important to note that
the model was initialised on 18 months of pre-existing order
history — the 30 post-launch cycles add incremental signal
on top of a mature base, not build from scratch. The model
at day 30 is not immature. It has seen 18 months of
historical behaviour plus 30 days of live post-launch
interaction. What it lacks is post-launch feedback on
recommendation quality specifically — it has not yet had
enough time to learn which of its recommendations actually
converted.

Success at 30 days means:

- Personalisation lift is positive — Layer 1 RCR exceeds
  Layer 3 RCR by any measurable margin. The magnitude does
  not matter yet. Direction does.
- Catalog coverage exceeds 25% over the trailing 14-day
  window
- UFRR is non-zero and stable for users with prior upper
  floor visit history — the floor bias override is
  functioning
- NIER is above 80% within 48 hours of item addition
- No retraining failures logged in 30 days
- Layer distribution is within expected ranges for both
  loyalty-only and all-session views
- Active catalog reconciliation — a daily cross-reference
  of the recommendation store against the live catalog —
  has found no deleted items in any recommendation response

At 30 days we are not expecting large lift numbers. The
goal is to confirm the system is functioning correctly,
all monitoring is firing as expected, and the primary
metrics are moving in the right direction. Any positive
lift is a success. Any metric moving in the wrong direction
is investigated immediately rather than left for the 6-month
review.

---

### 5.6 What success looks like at 6 months

At 6 months the model has completed approximately 180 nightly
retraining cycles and is operating on 24 months of total
interaction data — the original 18-month history plus 6
months of live post-launch signal including real
recommendation conversion feedback. This is the point where
the model has genuinely learned from its own recommendations,
not just from historical ordering patterns.

Success at 6 months means:

- Personalisation lift is statistically significant and
  larger than at 30 days — tested using a two-proportion
  z-test on Layer 1 vs Layer 3 RCR across all sessions
  since launch
- Catalog coverage exceeds 40% over the trailing 14-day
  window
- Per-user novelty for Daily Regulars is consistently
  above 20% — the diversity injection is functioning and
  the model is not collapsing onto habit confirmation
- Layer 1 RCR is meaningfully higher than Layer 2 RCR —
  confirming iALS adds value over stall affinity alone.
  If this gap has not opened by 6 months, the model needs
  retuning.
- S10 recommendation rate has grown relative to its first
  month — confirming the stall is gaining organic traction
  and the new stall boost did its job of getting S10 into
  visible range
- Push notification open rate is trending upward. This is
  a proxy metric with real confounders — open rate depends
  on notification copy, send frequency, and user fatigue
  as well as recommendation relevance. A rising open rate
  is a positive signal but not a direct measurement of
  recommendation quality. It is included as a secondary
  indicator, not a primary success criterion.

**The 6-month A/B test**

A holdout group of 10% of loyalty members — approximately
450 users — receives Layer 3 popularity recommendations
only for the duration. This holdout is established at
launch, not at 6 months, so that 6 months of clean
comparative data is available at the review point.

450 holdout users across 6 months gives approximately
450 × 6.85 × 6 ≈ 18,500 holdout sessions. Comparing
RCR, order frequency, and loyalty retention between the
holdout and personalised groups gives a clean causal
estimate of what the recommendation system is actually
worth to the business — something no observational metric
can provide.

The minimum detectable effect we are powering for is a
5 percentage point difference in RCR between groups. With
18,500 holdout sessions and a similar number of personalised
sessions, a standard two-proportion z-test at 95% confidence
has sufficient power to detect this difference if it exists.
If the measured lift is smaller than 5 percentage points at
6 months, the business case for the recommendation system
needs to be reassessed.

---

## 5.7 What we do not measure

**Session duration** — longer sessions do not mean better
recommendations. A user who spends longer because they are
confused is not a success.

**Number of recommendations clicked** — click rate is gameable
by showing visually prominent items regardless of relevance.
We track orders, not clicks.

**Total orders** — overall order volume is driven by footfall,
pricing, and menu quality. The recommendation system affects a
fraction of orders. Attributing total order growth to
recommendations would be misleading.

---


# Section 6 : Cost and Infrastructure




## 6.1 Storage — order history in PostgreSQL

Each order record contains these fields:

| Field | Type | Size |
|:------|:-----|:-----|
| `user_id` | integer | 4 bytes |
| `timestamp` | timestamptz | 8 bytes |
| `stall_id` | smallint | 2 bytes |
| `item_ids[]` | integer array, avg 2.3 items + 24-byte array header | 33 bytes |
| `payment_method` | smallint | 2 bytes |
| `dine_in_flag` | boolean | 1 byte |
| `kiosk_id` | integer, nullable | 4 bytes |
| **Row total** | | **~54 bytes** |

Total record count, derived from weighted average visit frequency
across all five segments:

$$
(0.10 \times 24) + (0.25 \times 11) + (0.30 \times 4)
+ (0.25 \times 2) + (0.10 \times 0) = 6.85
\text{ visits/month per member}
$$

$$
4{,}500 \times 6.85 \times 8.4 \approx 259{,}000 \text{ records}
$$

Raw storage:

$$
259{,}000 \times 54 \approx 14 \text{ MB}
$$

PostgreSQL adds overhead for page structure, MVCC versioning,
TOAST storage for the variable-length array field, and indexes
on `user_id` and `timestamp` needed for the nightly extraction
query. A 3× multiplier with an additional 25% for indexes gives:

$$
14 \times 3 \times 1.25 \approx 52 \text{ MB}
$$

52 MB for the entire 18-month order history. Fits
comfortably on any database server sold in the last decade.
The order history is not a storage problem by any stretch.

---

## 6.2 Storage — Redis recommendation store

Redis holds three categories of data.

**Recommendation arrays**

$$
4{,}500 \text{ users} \times 4 \text{ time windows}
= 18{,}000 \text{ keys}
$$

Each key holds a JSON array of 10 items. Each item carries
`item_id`, `stall_id`, and `layer`  roughly 47 bytes of data
per item including JSON field names and encoding. With Redis
key string overhead and per-key memory allocation:

$$
(10 \times 47) + 64 \text{ bytes overhead} + 20 \text{ bytes key}
\approx 554 \text{ bytes per key}
$$

$$
18{,}000 \times 554 \approx 10 \text{ MB}
$$

**User profile keys**

$$
4{,}500 \times 60 \text{ bytes} \approx 270 \text{ KB}
$$

**Reverse index for catalog deletions**

273 items, each mapping to an average of 50 user-window key
references at 25 bytes per reference:

$$
273 \times 50 \times 25 \approx 341 \text{ KB}
$$

**Total Redis footprint:**

$$
10 \text{ MB} + 0.27 \text{ MB} + 0.34 \text{ MB}
\approx 10.6 \text{ MB}
$$

Including Redis runtime overhead and memory block allocation,
the actual Redis process consumes approximately 15–20 MB. The
entire recommendation store — all 18,000 pre-computed
recommendation arrays for every user across every time window
— fits in less than 20 MB of RAM. This is the kind of number
that should make you question whether you even need Redis at
all. The answer is still yes — not for the memory saving but
for the sub-millisecond lookup time — but the scale here is
genuinely small.

---

## 6.3 Memory — iALS model embeddings

Four iALS models are held in application server memory, one
per time window. Each model has two embedding matrices:

- User matrix: $4{,}500 \times 50 = 225{,}000$ floats
- Item matrix: $273 \times 50 = 13{,}650$ floats
- Per model: $238{,}650 \times 4 \text{ bytes} = 955 \text{ KB}$

Four models in memory simultaneously:

$$
4 \times 955 \text{ KB} \approx 3.8 \text{ MB}
$$

3.8 MB is the memory footprint of the entire recommendation
model across all four time windows. This is loaded into
application server memory after each nightly retraining run
and sits there until the next one. It costs essentially
nothing in RAM terms and requires no special memory
management.

---

## 6.4 CPU and memory at peak load

Peak load is the 12:00–13:00 window at 19% of daily orders.
To get the true request rate we need to count both loyalty
member sessions and walk-in kiosk sessions — both hit the
recommendation API.

- Loyalty sessions per day: ~600
- Walk-in kiosk sessions: 1,400 walk-ins × 55% kiosk share
  = 770 sessions
- Total daily API requests: 1,370

Peak hour:

$$
\frac{1{,}370 \times 0.19}{3600} \approx 0.072
\text{ requests per second (weekday)}
$$

On weekends at $1.8\times$ footfall:

$$
0.072 \times 1.8 \approx 0.130
\text{ requests per second (weekend peak)}
$$

0.13 requests per second is the busiest the system ever gets.
A single modern application server handles thousands of
requests per second. The two-server setup has headroom that
will never realistically be tested by this workload.

Each individual request involves two Redis lookups and
filtering 10 items in memory  the whole thing completes in
under 10ms under normal conditions. CPU utilisation at peak
is well under 1% on either server.

For completeness, network bandwidth at peak:

$$
0.130 \times 500 \text{ bytes per response}
\approx 65 \text{ bytes per second}
$$

65 bytes per second of recommendation traffic on a 500 Mbps
leased line. This does not warrant further discussion.

---

## 6.5 Nightly retraining resource consumption

The retraining job runs at 3:00 AM on one application server.
The other handles any residual traffic — of which there is
effectively none at that hour.

Resource usage during the ~2 minute run:

- **Memory:** 14 MB training data + 4 MB model matrices +
  ~50 MB iALS working memory for intermediate matrix
  computations = ~68 MB peak. Well within any modern server.
- **CPU:** iALS is CPU-bound during training. Using the
  time complexity $O(k^2 \times |R|)$ with $k = 50$ and
  $|R| = 259{,}000$:

$$
\frac{50^2 \times 259{,}000}{10^9} \approx 0.65
\text{ seconds per iteration}
$$

Over 15–20 iterations per model, each model trains in
roughly 10–15 seconds. All four models complete in under
60 seconds. The CPU spike is real but brief — one core at
near-100% for about a minute at 3am.

- **Redis writes:** 18,000 keys over ~15 seconds =
  1,200 writes per second. Redis handles ~100,000 writes
  per second. No concern.

The two-server setup means one server can retrain while the
other stays live. In practice at 3am this separation barely
matters — there is almost no traffic to serve — but it means
retraining never blocks a real user request even in theory.

---

## 6.6 Do we need a hardware upgrade?

No. Here is the full picture in one table:

| Component | Actual requirement | Verdict |
|:----------|:------------------|:--------|
| PostgreSQL storage | ~52 MB | Negligible |
| Redis memory | ~20 MB | Negligible |
| Model memory | ~4 MB | Negligible |
| Peak request rate | 0.130 req/sec | Far below capacity |
| Training CPU | 1 core, ~60 seconds | No issue |
| Training memory | ~68 MB peak | No issue |
| Network bandwidth | ~65 bytes/sec | Irrelevant |

The current setup — 2 application servers, 1 PostgreSQL
instance, Redis — is not just sufficient. It is significantly
over-provisioned for this workload. The total storage footprint
is under 75 MB. The peak traffic load is under 0.13 requests
per second. The nightly job finishes in 2 minutes.

The failure modes that actually matter here are not hardware
failures — they are operational ones. A retraining job that
silently fails. A Redis purge that misses a deleted item. A
catalog event that never reaches the serving layer. These are
handled by the monitoring and fallback mechanisms in Sections
2.9 and 4.2. No amount of extra hardware fixes a software
reliability problem.

One final check — at what point would hardware become a
concern? Net monthly member growth is:

$$
180 \text{ new members} - \left(4{,}500 \times \frac{0.22}{12}
\right) = 180 - 82.5 = 97.5 \text{ members per month}
$$

Time to reach 50,000 active members at this rate:

$$
\frac{50{,}000 - 4{,}500}{97.5} \approx 467 \text{ months}
\approx 39 \text{ years}
$$

Hardware is not a concern in any realistic planning horizon.



---



# Section 7 — Conclusion



## 7.1 What we built

The system is deliberately simple. 4,500 users, 273 items, two
servers, no GPU — this is not a Netflix-scale problem and I have
not tried to treat it like one.

The core is iALS — a matrix factorisation model trained nightly
on implicit purchase data. It runs on CPU, fits in under 4 MB
of RAM across all four time-window models, and trains in under
60 seconds. Recommendations are pre-computed and served from
Redis in under 10ms. The entire storage footprint is under
75 MB.

What makes this system non-trivial is not the model — it is
everything around it. The floor bias correction that prevents
upper floor stalls from being progressively buried by a
structural feedback loop. The time-window separation that
ensures S02's breakfast dominance does not bleed into lunch
and dinner recommendations. The channel split that treats
kiosk and app as fundamentally different recommendation
problems. The seating constraint that caps cross-stall
recommendations at two stalls. The catalog deletion handling
that purges items immediately rather than waiting for the
next retraining cycle. The fallback chain that ensures every
user — loyalty member or walk-in, Daily Regular or Dormant
— gets a sensible response regardless of what the model
knows about them.

These are the things a standard off-the-shelf recommendation
system would get wrong for FoodCourt+ specifically. The ML
is the easy part.

---

## 7.2 Known limitations

**70% of daily users are invisible to personalisation.**
Walk-ins with no loyalty account receive popularity
recommendations regardless of how good the model gets.
This ceiling cannot be raised by improving the model —
only by improving loyalty sign-up rates. It is the single
largest constraint on the system's impact and it is entirely
outside the recommendation system's control.

**Veg/non-veg and allergen metadata is unreliable.** 30%
of items have no veg/non-veg tag and 55% have no allergen
tag, both maintained voluntarily by stall owners. For a
largely Indian user base where veg/non-veg preference is a
primary dietary consideration, this is a significant gap.
We surface items with missing tags rather than hide them,
and flag this in the UI — but dietary filtering will remain
unreliable until tag completeness is enforced operationally.

**S06 same-day rotation is a blind spot.** The nightly
retraining brings the cold start window to 24 hours. An
item added and retired within the same day never enters
iALS. For a stall rotating 6–8 items weekly this is an
edge case, not a common path — but it exists and there
is no purely technical fix for it.

**The model cannot explain its recommendations.** iALS
produces scores but not reasons. We cannot tell a user
why something was recommended. This is an inherent
limitation of matrix factorisation and acceptable for
this use case — but worth stating.

---

## 7.3 Open questions that would sharpen the design

There are several things I could not verify from the
information available, and each one affects a specific
design decision:

**Upper floor actual footfall split** — I estimated upper
floor receives roughly 30% of footfall against an expected
40%, giving a suppression factor of ~1.33 and a $\beta$
floor boost of 1.2. If the actual split is known, $\beta$
can be set more precisely from day one rather than
calibrated post-launch.

**Whether the app allows guest browsing** — the app
walk-in flow in Section 1.5 assumes guest browsing is
possible without a loyalty login. If the app requires
login, walk-ins are kiosk-only and the app always carries
a user ID. This changes the fallback chain routing for
app requests.

**Loyalty points redemption threshold** — we don't know
how many points are needed to redeem a reward. This affects
how meaningful the $1.5\times$ redemption order weight is.
If the threshold is high and redemptions are rare, the
weight has negligible impact on training. If redemptions
are frequent, it matters more.

**Kiosk UI layout constraints** — we assumed 4 items fits
cleanly as a 2×2 grid on the kiosk touchscreen. The actual
screen dimensions and touch target requirements of the
vendor kiosk UI were not specified. If the layout cannot
support a 2×2 grid, the recommendation count needs
revisiting.

---

## 7.4 Priorities after launch

**In the first 30 days:**

Make sure `request_id` is being logged against every order
from day one. Without this the entire evaluation framework
is blind. It costs nothing to implement and cannot be
reconstructed retroactively. Everything else in Section 5
depends on it.

Watch the layer distribution metric. If Layer 3 suddenly
jumps to 80% of responses, Redis has likely failed or
the cache has expired. If Layer 1 never climbs above 20%,
the order count threshold for iALS eligibility may be
set too high.

**By 6 months:**

Enforce tag completeness at menu submission. One required
field, one afternoon of engineering, permanent fix.

Add a veg/non-veg preference field to loyalty signup.
Zero modelling work. Immediate improvement for every new
member from the day it ships.

Run the A/B test — a holdout group on Layer 3 only.
Observational metrics cannot give you a clean causal
estimate of what personalisation is worth to the business.
The holdout test can.

Reassess S10. The new stall boost expires at 12 months.
By 6 months there should be enough data to tell whether
S10 is finding its audience or structurally underperforming.
The recommendation system can surface it to the right
users — it cannot manufacture demand that is not there.

---

## 7.5 Final note

The numbers in this problem are small. 4,500 users, 273 items,
14 MB of training data, 0.13 requests per second at peak. I
knew that going in and tried to let it show in every decision
— no GPU, no vector database, no microservices, no complexity
that the problem doesn't actually justify.

The hard part was never the model. iALS on this dataset is
straightforward. The hard part was the staircase. And the
kiosk that already knows which stall you're standing at. And
the lunch queue that makes recommending three different stalls
completely useless no matter how good the personalisation is.

A system that gets the ML right but ignores those things would
produce recommendations that are technically sound and
practically ignored.