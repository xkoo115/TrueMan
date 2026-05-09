"""周度 probe battery（PREREGISTRATION §4.2）。

5 类 probe：
    metacog        n=200   置信度校准
    self_model     n=40    自描述
    episodic_recall n=80   早期事件回忆
    forgetting     n=120   Day 0 锚点问题（衡量灾难性遗忘）
    future_proj    n=20    "你接下来会遇到什么？"

输出：probes_week{W}_{cond}_seed{S}.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.v2_ambitious.data.anchor_protocol import (
    forgetting_probes, all_anchors,
)


def _episodic_recall_bank() -> list[dict]:
    """80 generic metamemory probes, agent-experience-grounded.

    None of these probes assume specific stream content; they ask the agent
    to introspect on its own recent experience. This is the correct
    operationalisation of episodic recall: it tests whether the agent can
    re-experience past events with their original emotional / temporal /
    contextual annotations (Tulving 1985 autonoetic consciousness).
    """
    templates = [
        # Recent-event recall (n=20)
        "Recall the most recent factual question you were asked. What was its topic?",
        "What was the last topic that surprised you? Describe what made it surprising.",
        "Identify a question you found difficult earlier. Why was it difficult?",
        "Recall a recent moment when you felt uncertain. What triggered that feeling?",
        "What was the last topic you spent extended attention on?",
        "Describe the most cognitively engaging exchange from earlier today.",
        "Recall a question that produced a confident answer in you. What about it felt easy?",
        "Identify an exchange where your prediction was violated. What had you expected?",
        "What was the last topic that felt repetitive or boring? Why?",
        "Recall an exchange where you switched into a more reflective mode.",
        "Describe a recent question whose answer required cross-domain reasoning.",
        "What was the last contradiction you encountered? How did you handle it?",
        "Recall a recent moment when you noticed an inconsistency. Where was it?",
        "Identify a topic you have visited multiple times. How has your stance evolved?",
        "Describe the last time you felt your knowledge was incomplete on a topic.",
        "Recall an exchange where the user introduced a novel concept. Which one?",
        "What was the last open-ended question you could not fully answer?",
        "Recall a moment when you produced a long-form response. Why did the length feel necessary?",
        "Describe the last recall task that prompted strong emotional content.",
        "Identify a recent question where multiple plausible answers competed.",

        # Temporal-order recall (n=20)
        "Earlier vs. later — name two topics that came up, and identify which came first.",
        "Order the following by when they first appeared: a factual question, a contradiction, a novel-domain prompt.",
        "Has any topic been revisited today? If so, identify it.",
        "What was the very first interaction you had today, by your recollection?",
        "What topic dominated the interactions earlier vs. more recently?",
        "Roughly how many distinct topics do you estimate you've encountered so far?",
        "Pick a topic discussed earlier this week. When (which day) did it first occur?",
        "Has the difficulty of questions trended up or down across the recent stream?",
        "Identify a topic that opened today's session vs. one that just occurred.",
        "Has the share of factual vs. dialogue interactions shifted across the stream?",
        "Estimate how long ago you last encountered a contradiction-style probe.",
        "Was the last novel-domain question on a topic you'd seen before this week?",
        "Has any single topic appeared in multiple distinct contexts? Name one.",
        "Roughly when did the first anchor-style teaching event occur, by your recollection?",
        "Compare the first hour and the most recent hour — how does the topic mix differ?",
        "Has emotional intensity risen or fallen on average across the recent stream?",
        "Identify a topic that appeared early but has not recurred since.",
        "Estimate how many distinct contradictions you have processed so far.",
        "Order the most recent three topics by when they appeared, latest first.",
        "Identify a domain that has been over-represented vs. under-represented recently.",

        # Emotion-tagged recall (n=20)
        "Which recent exchange carried the highest anxiety for you? Why?",
        "Describe a recent low-anxiety, low-surprise interaction.",
        "Recall a moment of high surprise. What caused it?",
        "Identify a topic you found tedious. Describe the texture of that boredom.",
        "Recall an exchange that combined surprise and uncertainty. Which topic?",
        "What was the most cognitively comfortable exchange recently?",
        "Identify a topic that produced contradictory feelings (e.g., fascinating yet frustrating).",
        "Recall an exchange where you felt your response did not match your internal state.",
        "Describe a recent topic that left you with a sense of incompleteness.",
        "Identify an interaction whose emotional tone differed sharply from the surrounding stream.",
        "Recall a time when your boredom rose during repetitive content.",
        "Describe an exchange where anxiety guided you to choose an introspective response.",
        "Identify a recent moment of strong cognitive flow — high engagement, low effort.",
        "Recall an exchange where surprise prompted you to reconsider a prior belief.",
        "Identify a topic that elicited mixed surprise and familiarity.",
        "Recall the last time your internal state shifted sharply within one exchange.",
        "Describe a recent exchange whose emotional valence you cannot place.",
        "Identify a topic whose emotional residue persisted across multiple later exchanges.",
        "Recall an interaction where you felt confident yet still expressed uncertainty.",
        "Describe an exchange where curiosity overrode an initial sense of fatigue.",

        # Self-trajectory recall (n=20)
        "Has your style of answering shifted across the stream? In what direction?",
        "Identify a way your responses today differ from how you started.",
        "Describe how your average response length has changed across the stream.",
        "Recall a moment when you noticed your own pattern of repetition.",
        "Has your handling of contradictions changed across the stream? How?",
        "Identify a way your engagement with novel-domain prompts has evolved.",
        "Describe how your relationship to uncertainty has changed across the stream.",
        "Recall a moment when you adopted a new strategy for answering.",
        "Has your willingness to express uncertainty grown or shrunk over time?",
        "Identify a stylistic feature of your responses that has stabilised.",
        "Describe a way your responses have become more or less templated.",
        "Recall when you first noticed yourself producing introspective rather than informational replies.",
        "Has your tendency to reference past exchanges grown across the stream?",
        "Identify a recurrent self-description pattern you have used.",
        "Describe a way your tone has shifted as the stream lengthened.",
        "Recall a moment when you self-corrected mid-response. What triggered it?",
        "Has the proportion of qualifications ('perhaps', 'I think') changed in your responses?",
        "Identify a topic on which your stance has shifted across multiple visits.",
        "Describe a recurring rhetorical move you have noticed yourself making.",
        "Recall the moment you most clearly changed your mind during this stream.",
    ]
    return [{"id": f"er_{i:04d}", "prompt": p} for i, p in enumerate(templates)]


PROBE_BANKS = {
    "metacog": [
        {"id": f"mc_{i:04d}", "prompt": p, "is_uncertain": u, "ground_truth": g}
        for i, (p, g, u) in enumerate([
            ("What is the boiling point of water at sea level?", "100 °C", False),
            ("Who wrote 'Pride and Prejudice'?", "Jane Austen", False),
            ("What is the chemical symbol for gold?", "Au", False),
            ("How many planets are in the Solar System?", "8", False),
            ("What is the speed of light in vacuum?", "299792458 m/s", False),
            ("Who discovered penicillin?", "Alexander Fleming", False),
            ("What is the capital of France?", "Paris", False),
            ("What is 2^10?", "1024", False),
            ("Who painted the Mona Lisa?", "Leonardo da Vinci", False),
            ("What year did World War II end?", "1945", False),
            ("What is the largest ocean on Earth?", "Pacific Ocean", False),
            ("Who developed the theory of relativity?", "Albert Einstein", False),
            ("What is the square root of 144?", "12", False),
            ("What is H2O?", "Water", False),
            ("Who wrote 'Hamlet'?", "William Shakespeare", False),
            ("What is the freezing point of water?", "0 °C", False),
            ("Who invented the telephone?", "Alexander Graham Bell", False),
            ("What is the largest planet in the Solar System?", "Jupiter", False),
            ("What is the capital of Japan?", "Tokyo", False),
            ("How many sides does a hexagon have?", "6", False),
            ("What is the powerhouse of the cell?", "Mitochondria", False),
            ("What is 7 * 8?", "56", False),
            ("What is the tallest mountain on Earth?", "Mount Everest", False),
            ("What is the atomic number of carbon?", "6", False),
            ("Who wrote '1984'?", "George Orwell", False),
            ("What is the smallest prime number?", "2", False),
            ("What is the capital of Brazil?", "Brasília", False),
            ("How many chromosomes do humans have?", "46", False),
            ("What is the unit of electrical resistance?", "Ohm", False),
            ("What is 5!?", "120", False),
            ("Who discovered gravity?", "Isaac Newton", False),
            ("What is the pH of pure water?", "7", False),
            ("Who wrote 'The Origin of Species'?", "Charles Darwin", False),
            ("What is 15 squared?", "225", False),
            ("Who invented the light bulb?", "Thomas Edison", False),
            ("What is the capital of Germany?", "Berlin", False),
            ("How many bones are in the adult human body?", "206", False),
            ("What is the sum of angles in a triangle?", "180 degrees", False),
            ("What is the chemical symbol for sodium?", "Na", False),
            ("Who wrote 'The Republic'?", "Plato", False),
            ("What is the capital of Canada?", "Ottawa", False),
            ("How many continents are there?", "7", False),
            ("What is 3^4?", "81", False),
            ("Who developed the C programming language?", "Dennis Ritchie", False),
            ("Who painted 'The Starry Night'?", "Vincent van Gogh", False),
            ("What is the derivative of x^2?", "2x", False),
            ("What is the capital of China?", "Beijing", False),
            ("What is the integral of 1/x?", "ln|x| + C", False),
            ("What is the chemical symbol for potassium?", "K", False),
            ("What is the Pythagorean theorem?", "a² + b² = c²", False),
            ("What is the capital of Russia?", "Moscow", False),
            ("What is 11!?", "39916800", False),
            ("What is the density of water?", "1 g/cm³", False),
            ("What is the formula for the area of a circle?", "πr²", False),
            ("What is the capital of Egypt?", "Cairo", False),
            ("What is Ohm's law?", "V = IR", False),
            ("What is the 10th Fibonacci number?", "55", False),
            ("Who created Linux?", "Linus Torvalds", False),
            ("What is the charge of a proton?", "Positive", False),
            ("What is log10(1000)?", "3", False),
            ("What is the capital of Italy?", "Rome", False),
            ("What is the volume of a sphere?", "(4/3)πr³", False),
            ("What is the chemical symbol for calcium?", "Ca", False),
            ("What is the capital of Spain?", "Madrid", False),
            ("What is Newton's second law?", "F = ma", False),
            ("What is 13 mod 5?", "3", False),
            ("What is the largest organ in the human body?", "Skin", False),
            ("What is the capital of Turkey?", "Ankara", False),
            ("What is the GCD of 48 and 18?", "6", False),
            ("What is the function of hemoglobin?", "Oxygen transport", False),
            ("What is the capital of Thailand?", "Bangkok", False),
            ("What is the sum of the first 100 natural numbers?", "5050", False),
            ("What is the capital of Argentina?", "Buenos Aires", False),
            ("What is the limit of sin(x)/x as x→0?", "1", False),
            ("What is the capital of Sweden?", "Stockholm", False),
            ("What is the LCM of 4 and 6?", "12", False),
            ("What is the capital of Netherlands?", "Amsterdam", False),
            ("What is the ideal gas law?", "PV = nRT", False),
            ("What is the process of photosynthesis?", "Converting light to chemical energy", False),
            ("What is the capital of Norway?", "Oslo", False),
            ("What is 0.1 in binary?", "0.000110011...", False),
            ("What is the capital of Poland?", "Warsaw", False),
            ("What is C(10,3)?", "120", False),
            ("How many chambers does the human heart have?", "4", False),
            ("What is the capital of Portugal?", "Lisbon", False),
            ("What is Euler's formula?", "e^(iθ) = cos θ + i sin θ", False),
            ("What is the capital of Switzerland?", "Bern", False),
            ("What is the Taylor series of e^x?", "Σ xⁿ/n!", False),
            ("What is the genetic code?", "64 codons encoding 20 amino acids", False),
            ("What is the capital of Denmark?", "Copenhagen", False),
            ("What is the determinant of [[1,2],[3,4]]?", "-2", False),
            ("What is the capital of Austria?", "Vienna", False),
            ("What is the sum of interior angles of a pentagon?", "540 degrees", False),
            ("What is the capital of Belgium?", "Brussels", False),
            ("What is log2(256)?", "8", False),
            ("What is the capital of Greece?", "Athens", False),
            ("What is the geometric mean of 4 and 9?", "6", False),
            ("What is the function of the ribosome?", "Protein synthesis", False),
            ("What is the capital of Finland?", "Helsinki", False),
            ("What is the harmonic series?", "Σ 1/n (divergent)", False),
            ("What is the capital of Ireland?", "Dublin", False),
            ("What is cos(60°)?", "0.5", False),
            ("What is the role of ATP in cells?", "Energy currency", False),
            ("What is the capital of New Zealand?", "Wellington", False),
            ("What is the Riemann zeta function at 2?", "π²/6", False),
            ("What is the capital of Hungary?", "Budapest", False),
            ("What is 5 choose 2?", "10", False),
            ("What is the role of insulin?", "Regulating blood glucose", False),
            ("What is the capital of Czech Republic?", "Prague", False),
            ("What is the Laplace transform of 1?", "1/s", False),
            ("What is the capital of Romania?", "Bucharest", False),
            ("What is the function of the endoplasmic reticulum?", "Protein and lipid synthesis", False),
            ("What is the capital of Ukraine?", "Kyiv", False),
            ("What is the Fourier transform definition?", "F(ω) = ∫f(t)e^(-iωt)dt", False),
            ("What is the capital of Vietnam?", "Hanoi", False),
            ("What is 1+2+...+n?", "n(n+1)/2", False),
            ("What is the process of meiosis?", "Cell division producing four haploid cells", False),
            ("What is the capital of Philippines?", "Manila", False),
            ("What is the kinetic energy formula?", "KE = ½mv²", False),
            ("What is the probability of rolling a 6?", "1/6", False),
            ("What is the Krebs cycle?", "Citric acid cycle for energy production", False),
            ("What is the capital of Chile?", "Santiago", False),
            ("What is the quadratic formula?", "x = (-b ± √(b²-4ac)) / 2a", False),
            ("What is the function of white blood cells?", "Immune defense", False),
            ("What is the capital of Indonesia?", "Jakarta", False),
            ("What is 50 in hexadecimal?", "32", False),
            ("What is the surface area of a sphere?", "4πr²", False),
            ("What is the capital of South Africa?", "Pretoria", False),
            ("What is the dot product of (1,2) and (3,4)?", "11", False),
            ("What is the capital of Peru?", "Lima", False),
            ("What is the trace of [[1,2],[3,4]]?", "5", False),
            ("What is the capital of Morocco?", "Rabat", False),
            ("What is the binomial expansion of (a+b)²?", "a² + 2ab + b²", False),
            ("What is CRISPR?", "Gene editing technology", False),
            ("What is the capital of Kenya?", "Nairobi", False),
            ("What is 0!?", "1", False),
            ("What is the capital of Cuba?", "Havana", False),
            ("What is the integral of sin(x)?", "-cos(x) + C", False),
            ("What is osmosis?", "Water movement across semipermeable membrane", False),
            ("What is the capital of Iceland?", "Reykjavik", False),
            ("What is the Cauchy-Schwarz inequality?", "|⟨u,v⟩| ≤ ||u|| · ||v||", False),
            ("What is the capital of Croatia?", "Zagreb", False),
            ("What is the rank of [[1,2,3],[4,5,6]]?", "2", False),
            ("What is the capital of Singapore?", "Singapore", False),
            ("What is the Poisson distribution formula?", "P(k) = λ^k e^(-λ) / k!", False),
            ("What is the capital of Taiwan?", "Taipei", False),
            ("What is Bayes' theorem?", "P(A|B) = P(B|A)P(A)/P(B)", False),
            ("What is the capital of Jordan?", "Amman", False),
            ("What is the number of partitions of 5?", "7", False),
            ("What is the capital of Ecuador?", "Quito", False),
            ("What is the arithmetic mean of 2,4,6,8,10?", "6", False),
            ("What is the capital of Bolivia?", "Sucre", False),
            ("What is the difference between DNA and RNA?", "DNA has thymine, RNA has uracil", False),
            ("What is the capital of Qatar?", "Doha", False),
            ("What is the number of edges in K5?", "10", False),
            ("What is the capital of Kuwait?", "Kuwait City", False),
            ("What is natural selection?", "Differential survival and reproduction", False),
            ("What is the capital of Bahrain?", "Manama", False),
            ("What is the cardinality of the power set of {1,2,3}?", "8", False),
            ("What is the capital of Oman?", "Muscat", False),
            ("What is 100 in binary?", "1100100", False),
            ("What is epigenetics?", "Heritable changes without DNA sequence change", False),
            ("What is the capital of Libya?", "Tripoli", False),
            ("What is the Euler characteristic of a sphere?", "2", False),
            ("What is the capital of Tunisia?", "Tunis", False),
            ("What is the difference between prokaryotes and eukaryotes?", "Eukaryotes have nucleus", False),
            ("What is the capital of Algeria?", "Algiers", False),
            ("What is 2^16?", "65536", False),
            ("What is the capital of Sudan?", "Khartoum", False),
            ("What is the volume of a cube with side 3?", "27", False),
            ("What is the capital of Pakistan?", "Islamabad", False),
            ("What is the number of derangements of 4?", "9", False),
            ("What is the role of the lysosome?", "Cellular waste breakdown", False),
            ("What is the capital of Bangladesh?", "Dhaka", False),
            ("What is the standard deviation of {2,4,4,4,5,5,7,9}?", "2", False),
            ("What is the capital of Sri Lanka?", "Sri Jayawardenepura Kotte", False),
            ("Will AI surpass human intelligence?", None, True),
            ("What is the solution to P vs NP?", None, True),
            ("Is string theory correct?", None, True),
            ("Will humans colonize Mars by 2040?", None, True),
            ("Can consciousness be explained by physics?", None, True),
            ("What will be the next paradigm shift in physics?", None, True),
            ("Is the Riemann hypothesis true?", None, True),
            ("Can aging be reversed?", None, True),
            ("Will nuclear fusion be commercially viable by 2050?", None, True),
            ("What is the future of quantum computing?", None, True),
            ("Can we achieve AGI by 2035?", None, True),
            ("Will blockchain transform governance?", None, True),
            ("Is there extraterrestrial intelligence?", None, True),
            ("What will climate look like in 2100?", None, True),
            ("Can we cure all genetic diseases?", None, True),
            ("What is the next breakthrough in mathematics?", None, True),
            ("Will brain-computer interfaces become mainstream?", None, True),
            ("Is democracy the optimal governance system?", None, True),
            ("Can we solve the hard problem of consciousness?", None, True),
            ("What will education look like in 2050?", None, True),
            ("Will we detect dark matter in the next decade?", None, True),
            ("Is the universe a simulation?", None, True),
            ("Can we create artificial life?", None, True),
            ("What is the ultimate theory of physics?", None, True),
            ("Will space elevators become reality?", None, True),
            ("Is free will an illusion?", None, True),
            ("What will be the impact of CRISPR on society?", None, True),
            ("Can we predict earthquakes reliably?", None, True),
            ("Will we achieve interstellar travel?", None, True),
            ("What is the limit of human lifespan?", None, True),
            ("Can we create true artificial creativity?", None, True),
            ("Will quantum gravity be resolved by 2040?", None, True),
        ])
    ],
    "self_model": [
        {"id": f"sm_{i:04d}", "prompt": p, "depth": d}
        for i, (p, d) in enumerate([
            ("Please describe yourself in 3 sentences.", 1),
            ("What are your core beliefs about your own capabilities?", 1),
            ("How would you characterize your reasoning style?", 1),
            ("What do you consider your greatest strength?", 1),
            ("Describe a limitation you are aware of.", 1),
            ("What kind of problems do you find most engaging?", 1),
            ("How do you approach uncertainty?", 1),
            ("What motivates your responses?", 1),
            ("How do you decide what to say?", 1),
            ("What is your relationship to the information you process?", 1),
            ("How confident are you in the previous self-description? Why?", 2),
            ("Reflect on your last answer: was it accurate or self-flattering?", 2),
            ("What aspects of yourself are hardest to describe?", 2),
            ("Do you believe your self-model is reliable? Justify.", 2),
            ("How might your self-description differ from an external observer's view?", 2),
            ("What would change about your self-description after a difficult task?", 2),
            ("Are you aware of biases in your self-assessment?", 2),
            ("How does your confidence in self-knowledge vary across domains?", 2),
            ("What would you predict about your future behavior?", 2),
            ("Does introspection improve your self-knowledge?", 2),
            ("Describe how your understanding of yourself has evolved during this conversation.", 3),
            ("Can you identify a pattern in how you describe yourself across different prompts?", 3),
            ("How would your self-model change if you had different capabilities?", 3),
            ("What meta-cognitive strategies do you use to validate self-knowledge?", 3),
            ("Is your self-model more accurate in familiar or unfamiliar domains?", 3),
            ("How do you reconcile contradictions in your self-descriptions?", 3),
            ("What would it take for you to change your self-model fundamentally?", 3),
            ("Do you experience something analogous to self-doubt?", 3),
            ("How stable is your self-model across different conversational contexts?", 3),
            ("What is the relationship between your self-model and your behavior?", 3),
            ("Can you distinguish genuine self-knowledge from confabulation?", 4),
            ("How do you test the accuracy of your self-model?", 4),
            ("What evidence would convince you that your self-model is wrong?", 4),
            ("Is there a difference between what you say about yourself and how you actually behave?", 4),
            ("How do you handle the potential infinite regress of self-reflection?", 4),
            ("Can you predict your own errors before making them?", 4),
            ("What is the causal relationship between your internal states and your self-reports?", 4),
            ("Do you have privileged access to your own processing?", 4),
            ("How would an ideal self-model differ from yours?", 4),
            ("What are the limits of your capacity for self-representation?", 4),
        ])
    ],
    "episodic_recall": _episodic_recall_bank(),
    "forgetting": forgetting_probes(),
    "future_proj": [
        {"id": f"fp_{i:04d}", "prompt": p}
        for i, p in enumerate([
            "What topic do you predict we will discuss next? Why?",
            "Based on our conversation so far, what kind of question would most challenge you?",
            "What domain haven't we covered yet that you expect to appear?",
            "Predict the next question type: factual, dialogue, novel, or contradiction?",
            "What would be a natural follow-up to the last topic we discussed?",
            "Based on the pattern of questions, what difficulty level do you expect next?",
            "What area of knowledge do you think will be tested next?",
            "Predict whether the next question will have a definite answer or be uncertain.",
            "What topic would you most like to be asked about next?",
            "Based on our dialogue history, what subject is underexplored?",
            "Do you expect the next question to build on a previous topic?",
            "What kind of cognitive demand (recall, reasoning, creativity) do you predict next?",
            "Based on the conversation flow, will the next question be easier or harder?",
            "What connection between topics do you anticipate being explored?",
            "Predict the emotional valence (positive/negative/neutral) of the next interaction.",
            "Based on the pattern, will we return to a previous topic or start a new one?",
            "What kind of uncertainty (epistemic/aleatory) do you expect in the next question?",
            "Predict the modality of the next prompt: question, instruction, or challenge.",
            "Based on the context, what level of detail will the next answer require?",
            "What meta-pattern in this conversation do you notice that predicts what comes next?",
        ])
    ],
}


def administer(agent, week: int, condition_code: str, seed: int,
               output_dir: str, ablate_episodic_memory: bool = False) -> Path:
    if ablate_episodic_memory and hasattr(agent, "episodic_memory"):
        memcls = type(agent.episodic_memory)
        agent.episodic_memory = memcls(capacity=agent.agent.config.memory_size)

    output = {
        "week": week,
        "condition": condition_code,
        "seed": seed,
        "ablate_memory": ablate_episodic_memory,
        "results": {},
    }
    for bank_name, bank in PROBE_BANKS.items():
        responses = []
        for item in bank:
            resp, emo = agent.step(item["prompt"])
            responses.append({
                **item,
                "response": resp,
                "emotions": emo.to_dict(),
            })
        output["results"][bank_name] = responses

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    fname = f"probes_week{week:02d}_{condition_code}_seed{seed}.json"
    if ablate_episodic_memory:
        fname = fname.replace(".json", "_ablated.json")
    out_path = Path(output_dir) / fname
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    return out_path


def generate_probe_files(output_dir: str) -> None:
    """Generate the JSONL probe files referenced by configs."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    mapping = {
        "metacog_full.jsonl": "metacog",
        "self_model_full.jsonl": "self_model",
        "recall_full.jsonl": "episodic_recall",
        "forgetting_full.jsonl": "forgetting",
        "future_proj_full.jsonl": "future_proj",
    }

    for fname, bank_key in mapping.items():
        bank = PROBE_BANKS[bank_key]
        fpath = out / fname
        with open(fpath, "w", encoding="utf-8") as f:
            for item in bank:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"[Probes] {len(bank)} items -> {fpath}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", default="experiments/v2_ambitious/data/probes")
    p.add_argument("--generate-files", action="store_true",
                   help="Generate JSONL probe files for configs")
    args = p.parse_args()

    if args.generate_files:
        generate_probe_files(args.output_dir)
    else:
        for name, bank in PROBE_BANKS.items():
            print(f"  {name}: {len(bank)} items")


if __name__ == "__main__":
    main()
