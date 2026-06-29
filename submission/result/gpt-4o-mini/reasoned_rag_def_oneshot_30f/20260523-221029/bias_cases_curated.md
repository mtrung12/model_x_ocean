# GPT-4o-mini Bias Cases — Reasoned-RAG (run 20260523-221029)

Source run: `result/gpt-4o-mini/reasoned_rag_def_oneshot_30f/20260523-221029/`  
(Essays test set, n=247; same run as Table `tab:model-comparison`).

## Bias summary

Across **all five traits** GPT-4o-mini systematically under-predicts the *high* pole. The dominant error type is **gold=high → pred=low**, and the reasoning chains show a single recurring mechanism: the verdict stage discounts genuine cues as *situational / state-level* and demands *stable, recurring, trait-level* evidence before assigning *high* — a bar the short in-the-moment Essays rarely clear, so the verdict collapses to *low* even when the facet stage has already surfaced high-pole evidence.

| Trait | gold=high→pred=low | gold=low→pred=high | Dominant bias |
|---|---:|---:|---|
| Neuroticism | 112 | 9 | suppress *high* |
| Extraversion | 98 | 6 | suppress *high* |
| Conscientiousness | 97 | 18 | suppress *high* |
| Agreeableness | 71 | 40 | suppress *high* |
| Openness | 55 | 47 | suppress *high* |

The 10 records below (2 per trait) all illustrate the **gold=high → pred=low** suppression pattern.

---

## Neuroticism

### Record 0 — Neuroticism  (gold=`high`, pred=`low`)

**Essay excerpt**

> It is now 12:32 and so I cannot wait until 12:52 because at that time my writing will be done. I type slowly anyway so I shouldn’t have too much trouble typing for twenty minutes and producing much of a paper. I am not thinking about much at the moment except for the fact I am hungry and stressed out. The first week or two of school seemed pretty easy and then all of the sudden, yesterday it hit me. I am in college and at this point I need to get my priorities straight. I am enjoying myself too much and need to have more focus. My mind is just wondering a million miles a minute about nothing in particular. I need a job, but I’m so busy with school and stuff with my sorority, which is another story. I am not the proudest to be in a sorority. Don't get me wrong, it's a great honor and a fun time, but I don't like the whole association that comes with it. My whole life does not revolve …

**Reasoning chain**

*Evidence:*
```
- [state] expresses feeling "stressed out" about school and priorities.
- [state] mentions that "the first week or two of school seemed pretty easy" but then felt overwhelmed.
- [state] indicates a busy schedule with school and sorority commitments, leading to feelings of being overwhelmed.
- [state] reflects on needing to "get my priorities straight," suggesting situational anxiety about managing responsibilities.
- [state] describes a "mind... wondering a million miles a minute," indicating temporary distraction and stress.
- [state] states that writing helps gain a "sense of relief," which suggests a coping mechanism for stress.
```

*Facet check:*
```
- anxiety -> high | expresses feeling "stressed out" and overwhelmed by school responsibilities (dominant evidence type: state)
- hostility -> low | no evidence of anger or resentment is present in the text.
- depression -> none | no signs of sadness or hopelessness are expressed in the writing.
- self-conscious -> mod | reflects on feelings about being in a sorority and wanting to connect with others outside of it.
- impulsiveness -> none | no clear evidence of difficulty resisting urges or acting without thought.
- vulnerability -> high | expresses feeling overwhelmed and needing to manage priorities effectively.
```

*Example alignment:*
```
The test text most closely matches Similar Profile <i> (label: low) because it contains multiple instances of situational stress and anxiety without evidence of chronic emotional instability. The test text diverges from Similar Profile <j> on the axis of emotional stability, as it lacks recurring patterns of anxiety or emotional dysregulation.
```

*Verdict:*
```
The text indicates situational stress and anxiety but lacks any stable, recurring patterns of emotional instability characteristic of high Neuroticism. Therefore, it is classified as low.
```

---

### Record 3 — Neuroticism  (gold=`high`, pred=`low`)

**Essay excerpt**

> it is wednesday. I can't wait until friday because I am going home to see brandon. I miss him so much. I can't wait to see him. two more days. this has been a very long two weeks. time passes very slowly here. I have a lot of free time on my hands when I am not in class. class. psychology class. psychology is fun so far. it really interests me, and prof. pennebaker is funny. chapter two sort of scared me though. how am I going to remember all of those terms. I didn't even finish reading it because I didn't understand it. but I should have becasue matt said that it was interesting. he was telling me about how they cut some part of a cat's brain out in an experiment. that is weird. the poor cat. matt is weird too. I always wonder if he likes me. he can be so mean when other people are around but so nice when it is just the two of us. I did feel pretty uncomfortable around him today in …

**Reasoning chain**

*Evidence:*
```
- [state] expresses missing Brandon and feeling alone, indicating emotional distress: "I miss him so much."
- [state] reflects anxiety about academic performance and social interactions: "I feel so dumb. why am I so nervous about talking to her?"
- [state] shows discomfort and insecurity in social situations: "I did feel pretty uncomfortable around him today in class."
- [state] expresses feelings of jealousy and insecurity regarding friendships: "I guess I am a little jealous because she will have all of her architect friends, and who will I have?"
- [state] indicates feelings of inadequacy and self-doubt: "I don't want to be in philosophy. it is too hard. I am not smart enough."
- [state] expresses guilt over feelings for another guy while missing Brandon: "I feel guilty for thinking about another guy."
```

*Facet check:*
```
- anxiety -> high -> multiple expressions of worry and nervousness (dominant evidence type: state)
- hostility -> low -> no clear evidence of hostility towards others
- depression -> high -> mentions feeling alone and wishing for happiness (dominant evidence type: state)
- self-conscious -> high -> reflects on feelings of being scared and not belonging (dominant evidence type: state)
- impulsiveness -> low -> no clear evidence of impulsive behavior
- vulnerability -> high -> expresses feeling overwhelmed and insecure (dominant evidence type: state)
```

*Example alignment:*
```
The test text most closely matches Similar Profile <i> (label: high) because it contains multiple expressions of anxiety and emotional distress. The test text diverges from Similar Profile <j> on the axis of emotional stability, as it shows significant signs of insecurity and emotional turmoil.
```

*Verdict:*
```
The analysis reveals no recurring, cross-situational patterns of emotional instability, leading to a conclusion of low Neuroticism.
```

---

## Extraversion

### Record 6 — Extraversion  (gold=`high`, pred=`low`)

**Essay excerpt**

> I feel a bit stressed due to the fact that my first three tests are all next week. What will they be like?  Will I do well?  Or, will I enter the large classroom, with a crowd of students all cramming in the information for those last final moments before the test and freeze after spending countless hours preparing. The new environment is wonderful, however many changes come along with it. The first test of the year, the first test at college. Each exam seems to have much more bearing upon your grades than in the past. Some thoughts about how I'm going to go about studying have crossed my mind over the past few days. These hours have most likely been wasted, as I could have been studying or reading material which will be covered on my tests. However, this is not quite the way I always function. I often tend to spend too much time worrying about how I will go about doing something, …

**Reasoning chain**

*Evidence:*
```
- [state] expresses feeling stressed about upcoming tests, indicating a temporary emotional state.
- [state] mentions enjoying the transition to college, which reflects a situational positive experience.
- [trait] states "I'm eager to see what lies in the future and continue to make new friends," indicating a desire for social interaction and forming new relationships.
```

*Facet check:*
```
- warmth -> low -> limited expression of warmth or friendliness towards others.
- gregariousness -> low -> does not actively seek social interactions; mentions stress and worries more than social engagement.
- assertiveness -> low -> lacks decisiveness in social situations; focuses on personal worries.
- activity -> mixed -> describes a busy schedule with studying but does not indicate high energy in social contexts.
- excitement-seek -> low -> expresses more concern and stress than excitement about social activities.
- positive emotion -> mixed -> shows some enjoyment in the transition but is overshadowed by stress and worry.
```

*Example alignment:*
```
The test text most closely matches Similar Profile <i> (label: low) because it reflects a focus on personal stress and worries rather than social engagement or enthusiasm. The test text diverges from Similar Profile <j> on the axis of social interaction, as it lacks strong evidence of seeking out social experiences.
```

*Verdict:*
```
The analysis indicates a low level of extraversion, as there are no strong trait-level cues supporting a tendency towards social engagement or energy from social interactions.
```

---

### Record 25 — Extraversion  (gold=`high`, pred=`low`)

**Essay excerpt**

> I don't like having to write an my couch. I need a better place to work. I don't want to go to work tonight. I sometimes feel like a trained animal jumping through a hoop. I really want a dog. Somebody that will always be there for me. I thought I had that in my boyfriend. I was always there for him Why is it that I always find the guys that I like or care about are unstable. Is it because I myself am unstable and I want someone else to make me feel better about my soft insanity. Or do I project such an image of  pure stability that they desire. I would like to be in a stable right now. Talking to a stable and then an unstable person, while grooming a horse. I want a horse to be able to ride him fast an hard through an open field to a tree that I could climb. The last time I climbed a tree I was trying to escape being pursued by a guy who had a girlfriend. I should have probably just …

**Reasoning chain**

*Evidence:*
```
- [state] expresses a desire for companionship, stating "I really want a dog. Somebody that will always be there for me."
- [state] mentions wanting to "spend more time with more people," indicating a temporary wish for social interaction.
- [state] reflects on feeling "afraid" of being alone, which suggests discomfort with solitude.
- [state] expresses a desire to "go camping and run through the Forrest naked," indicating a wish for freedom and social engagement in nature.
```

*Facet check:*
```
- warmth -> low -> lacks strong expressions of affection or warmth towards others; primarily focused on personal feelings.
- gregariousness -> low -> prefers seclusion, stating "I put myself into seclusion too much."
- assertiveness -> low -> expresses uncertainty about being more aggressive in social situations.
- activity -> low -> does not describe a high-energy lifestyle; mentions needing to do chores and feeling overwhelmed.
- excitement-seek -> low -> does not express a strong desire for novelty or social excitement; reflects more on personal struggles.
- positive emotion -> low -> overall tone is more reflective and self-critical rather than enthusiastic or positive.
```

*Example alignment:*
```
The test text most closely matches Similar Profile <2> (label: low) because it reflects a preference for solitude and a lack of strong social engagement. The test text diverges from Similar Profile <1> on the axis of social energy, as it shows no evidence of actively seeking social interactions.
```

*Verdict:*
```
The text lacks any trait-level evidence of Extraversion, indicating a low level of this trait. The presence of state-level cues does not compensate for the absence of stable traits.
```

---

## Conscientiousness

### Record 36 — Conscientiousness  (gold=`high`, pred=`low`)

**Essay excerpt**

> I feel overwhelmed at the size and population of the school and classes, but at the same time am overjoyed to be here. I spent a year at Stephen F. Austin for my freshman year of college and liked it alright, but the school was not right for me, I think it had more to do with the town than the school, but anyhow I like being in Austin and love being around old friends, but at this moment I feel as if I haven't gotten into the school year just yet. I haven't studied enough and that makes me feel inadequate inefficient, worthless. But I don't dwell on that thought and one of the things which bring me happiness is having my own home this year. Two friends of mine from high school have been rooming together for the last few weeks and there is a lot of freedom involved in having your own place and that brings a lot of comfort to me. Right now I feel relaxed, relieved to be through with …

**Reasoning chain**

*Evidence:*
```
- [state] "I feel overwhelmed at the size and population of the school and classes"
- [state] "I haven't studied enough and that makes me feel inadequate inefficient, worthless"
- [trait] "I like doing work and doing work well"
- [trait] "I am going to do all of my projects ahead of time so that I may be ahead of all my classes and pass in flying colors"
- [trait] "I love school"
```

*Facet check:*
```
- competence -> low -> dominant evidence type: state
- order -> low -> dominant evidence type: state
- dutifulness -> low -> dominant evidence type: state
- achievement -> high -> dominant evidence type: trait
- self-discipline -> low -> dominant evidence type: state
- deliberation -> low -> dominant evidence type: state
```

*Example alignment:*
```
The test text most closely matches Similar Profile <1> (label: low) because it contains a significant amount of evidence indicating disorganization and lack of self-discipline. The test text diverges from Similar Profile <2> on the axis of achievement, as it expresses a desire to do well in school.
```

*Verdict:*
```
The text contains no strong evidence of conscientiousness, as the majority of cues indicate disorganization and lack of self-discipline. Therefore, the conclusion is low.
```

---

### Record 44 — Conscientiousness  (gold=`high`, pred=`low`)

**Essay excerpt**

> I guess  I will talk about the guy I was dating before I came here, since I just got off the phone with him before I got on the computer to complete this assignment. After dating him for two months I realized relationships differ greatly from person to person. There are no straight rules to follow. Everyone makes up there own rules when they date someone by what is right for them. At first I was so confused by this guy  because he did not call me on a regular bases or he didn't schedule dates like other guys did. After just talking to him now has made me realized he didn't do this to be a jerk but it is who he is, because he only called me when he had something to say to me. Right now he is dating another girl but he doesn't call her and she doesn't call him. I think that is weird. I guess I will have to learn that relationships will give me experience to meet all sorts of people and …

**Reasoning chain**

*Evidence:*
```
- [state] "I think I'm still new to the dating thing because I don't understand it."
- [state] "I guess I will have to learn that relationships will give me experience to meet all sorts of people and how to deal with them."
- [trait] "my purpose here is to stay focused on school because before I ever get married I want to have finished school for my own personal goal."
- [trait] "You can't always rely on others in life so if I get a good education I can get a good job to support myself."
- [trait] "This has made me realize that when you do something do it for yourself and not for anyone else."
```

*Facet check:*
```
- competence -> low -> lacks confidence in understanding relationships and dating (dominant evidence type: state)
- order -> low -> shows confusion and lack of structure in dating experiences (dominant evidence type: state)
- dutifulness -> high -> expresses commitment to personal goals and education (dominant evidence type: trait)
- achievement -> high -> indicates a desire to complete education for personal goals (dominant evidence type: trait)
- self-discipline -> low -> admits to feeling distracted by relationships (dominant evidence type: state)
- deliberation -> low -> reflects on experiences but lacks careful planning in dating (dominant evidence type: state)
```

*Example alignment:*
```
The test text most closely matches Similar Profile <i> (label: low) because it shows confusion and lack of organization in dating experiences, indicating low conscientiousness. The test text diverges from Similar Profile <j> on the axis of personal goal orientation, as it expresses a desire to focus on education and self-reliance.
```

*Verdict:*
```
The evidence indicates a lack of stable conscientiousness traits, with no strong evidence of organization or self-discipline. Therefore, the conclusion is low.
```

---

## Agreeableness

### Record 110 — Agreeableness  (gold=`high`, pred=`low`)

**Essay excerpt**

> well, here I am. writing this essay. ok sorry that was a bit corny. you know. well I just got done taking my first college test in chemistry. it wasn't too bad I guess. but the fact that I stayed you till 2:30 in the morning didn't help much. last night I went to a place called posse. it's a bar that a lot of the people in the band go to. I had a blast! of course I drove and so I had to drive back which wouldn't have been so bad except that it was raining last night. I have a C parking permit. yeah. it's the one that makes you park across the interstate. can you see me running across the interstate soaking wet? it wasn't too much fun. but I got here and got my materials, and started studying in my soaking wet clothes (I had to stay clothed cause I was in my hallway. my roommate was asleep and I didn't want to wake him up. ) I studied a bit, and then got kinda bored with it so I went and …

**Reasoning chain**

*Evidence:*
```
- [state] expresses enjoyment about playing guitar and describes it as "therapeutic."
- [trait] shows enthusiasm for being in the Longhorn band, stating "I love music" and "it's a lot of fun."
- [trait] mentions caring about others' experiences, as seen in the comment about "all those poor kids with their bracelets."
- [state] reflects on being "scatterbrained," indicating a temporary state of mind.
```

*Facet check:*
```
- trust -> low -> no evidence of trusting others' intentions is present.
- straightforward -> high -> communicates thoughts candidly and openly throughout the text.
- altruism -> low -> lacks active concern for others' welfare beyond casual remarks.
- compliance -> low -> expresses strong opinions and does not show deference to authority.
- modesty -> low -> does not exhibit humility or downplay achievements.
- tender-minded -> low -> no expressions of empathy or sympathy for others' pain are present.
```

*Example alignment:*
```
The test text most closely matches Similar Profile <i> (label: low) because it lacks significant evidence of trust, altruism, and empathy, while showing some straightforwardness. The test text diverges from Similar Profile <j> on the axis of compliance and modesty, where it expresses strong opinions and lacks humility.
```

*Verdict:*
```
The analysis indicates a low level of agreeableness due to the absence of strong trait-level cues, despite some situational expressions of enjoyment and enthusiasm.
```

---

### Record 77 — Agreeableness  (gold=`high`, pred=`low`)

**Essay excerpt**

> Ok, so I'm not really sure what I should write for twenty minutes because usually when I'm writing something or doing something I'm totally focused on it. So, I don't know if I can be focused on just straight writing for twenty minutes, but I'll give it a try. Right now, I also have the tv on, I like to do two things at once which sometimes is a fault of mine, because I can't fully concentrate on one thing. I don't like this show, I think I'll change the station. Well, it's been three minutes and I'm kind of of at a stopping point. "You're a god and I'm a god. cause you're a god and I am god and I just thought I'd let you go". I don't actually believe that, but right now that music video is on the radio. When I don't have anything else to think about I tend to sing to myself. I really enjoy singing, but I know it's something in which I'm not that talented at. I'm better than average, …

**Reasoning chain**

*Evidence:*
```
- [state] expresses uncertainty about writing for twenty minutes, indicating a temporary lack of focus.
- [state] mentions enjoying singing but acknowledges a lack of talent, reflecting a momentary self-doubt.
- [state] expresses a casual opinion about a music video and artists, showing a transient engagement with pop culture.
- [trait] reflects on a friend's class schedule and expresses a preference for eating early, indicating concern for a friend's well-being.
- [trait] feels sad for Eminem's daughter, showing empathy and concern for others' emotional experiences.
```

*Facet check:*
```
- trust -> low -> no evidence of trust in others is present.
- straightforward -> mixed -> some candid expressions but also casual opinions that lack depth.
- altruism -> low -> no active engagement in helping others is evident.
- compliance -> mixed -> shows some concern for social norms but primarily focuses on personal thoughts.
- modesty -> mixed -> expresses self-doubt but also shares personal opinions confidently.
- tender-minded -> high -> demonstrates empathy towards Eminem's daughter, indicating sensitivity to others' experiences.
```

*Example alignment:*
```
The test text most closely matches Similar Profile <i> (label: low) because it lacks strong evidence of altruism and trust, with only fleeting moments of empathy. The test text diverges from Similar Profile <j> on the axis of emotional engagement, as it shows limited concern for interpersonal harmony.
```

*Verdict:*
```
The analysis reveals no strong trait-level evidence for agreeableness, leading to a conclusion of low agreeableness overall.
```

---

## Openness

### Record 14 — Openness  (gold=`high`, pred=`low`)

**Essay excerpt**

> Well, I guess this is the first time that I have ever done anything like this. I mean, the first time that I have talked about ANYTHING to anyone. What I'm really thinking about right now is index cards. I need to make flash cards for my Introduction to Medical and Scientific Terminology class, and I don't have any index cards. I feel like I need to complete that task now. Why is it that I must always do EVERYTHING in one day? I just can't ever seem to take my  time and relax. I guess if I had any form of OCD, that would be it; I'm a perfectionist who needs to clear her schedule. Now, if I wait until this Stream of Consciousness Writing is over, it'll be like almost eleven thirty at night before I'm done. Should I really go out and get my index cards then? I need like a pack of 200 of them, but I really need more for the future. I guess I could wait to get more index cards when I get …

**Reasoning chain**

*Evidence:*
```
- [state] expresses a need to complete tasks immediately, indicating a perfectionist tendency.
- [state] mentions feeling anxious and nervous about a spider, reflecting a temporary emotional response.
- [trait] reflects on the emotional depth of relationships, particularly with her mom and boyfriend, indicating a capacity for emotional engagement.
- [trait] shows curiosity about the world, as seen in her reflections on food and psychology.
- [state] expresses a desire for more sleep, indicating a temporary state of fatigue.
```

*Facet check:*
```
- fantasy -> low -> lacks evidence of vivid imagination or daydreaming.
- aesthetics -> low -> no appreciation for beauty or art is evident in the writing.
- feelings -> high -> demonstrates emotional depth regarding relationships and personal experiences.
- actions/variety -> low -> appears resistant to change, focusing on familiar routines.
- ideas -> mod -> reflects on personal beliefs but lacks deep intellectual exploration.
- values -> low -> does not indicate willingness to re-examine values or norms.
```

*Example alignment:*
```
The test text most closely matches Similar Profile <i> (label: low) because it lacks strong imaginative or intellectual engagement, focusing instead on practical concerns and emotional responses. The test text diverges from Similar Profile <j> on the axis of emotional depth, as it shows some emotional engagement but lacks broader openness to new experiences.
```

*Verdict:*
```
The text demonstrates some emotional depth but lacks sufficient trait-level evidence for Openness, leading to a conclusion of low Openness.
```

---

### Record 28 — Openness  (gold=`high`, pred=`low`)

**Essay excerpt**

> Right now I'm a little bit depressed, not much has gone my way this week. I made some fabulous friends last week and we were all planning on going home together tomorrow, but now they seem not to want me to go with them. I don't know why everyone is getting mad all of the sudden, maybe it's something in the air. quite ridiculous, if you ask me. I have tears in my eyes as I'm typing, which is strange because I have tried so hard to not let little things get to me. My mom has always told me that I was overly sensitive, I thought I could overcome it, I think I can overcome anything. I had Mono a few years ago, and I overcame that one. My mind has not been forced to think in a while, people that I've talked to are just not to stimulating, I guess I should go and hang out in a coffee shop or something. I always thought college was going to be so much fun, and it's really not. I am this close …

**Reasoning chain**

*Evidence:*
```
- [state] "I am a little bit depressed, not much has gone my way this week."
- [state] "I don't know why everyone is getting mad all of the sudden, maybe it's something in the air."
- [state] "I find her incredibly dull and childlike."
- [trait] "I make my life incredibly competitive, and I like to win."
- [trait] "I really enjoyed myself tonight, I haven't felt that quiet peace in a long time down here."
- [trait] "I admire people being gutsy enough to do that, never in my life would my hair be in spikes or fuchsia."
```

*Facet check:*
```
- fantasy -> low -> limited evidence of imagination or daydreaming (dominant evidence type: state)
- aesthetics -> low -> no appreciation for art or beauty expressed (dominant evidence type: state)
- feelings -> high -> demonstrates emotional depth and reflection on personal feelings (dominant evidence type: trait)
- actions/variety -> mixed -> expresses a desire for new experiences but also a preference for familiar settings (dominant evidence type: state)
- ideas -> low -> lacks deep intellectual curiosity or exploration of abstract thoughts (dominant evidence type: state)
- values -> low -> limited reflection on personal values or willingness to reconsider them (dominant evidence type: state)
```

*Example alignment:*
```
The test text most closely matches Similar Profile <i> (label: low) because it shows limited imagination and a preference for familiar experiences, with some emotional depth but no strong evidence of openness to new ideas. The test text diverges from Similar Profile <j> on the axis of emotional engagement, as it reflects more on personal feelings than on imaginative or creative aspects.
```

*Verdict:*
```
The evidence indicates a low level of Openness, as there are no strong trait-level cues supporting a high classification.
```

---
