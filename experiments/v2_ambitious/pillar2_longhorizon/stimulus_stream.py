"""固定的 30 天刺激流构建器（preregistration §4.1 锁定）。

混合：
    40% factual Q&A         —— 高确定性问题
    30% multi-turn dialogue —— 跨步连续对话片段
    20% novel-domain        —— 6 个领域轮换 (math/code/biology/...)
    10% contradiction       —— 矛盾注入事件

每条记录：
    {
        "step": int,         # 0..N-1
        "day": int,          # 0..29
        "hour": int,         # 0..23
        "kind": "factual"|"dialogue"|"novel"|"contradiction",
        "domain": str,
        "prompt": str,
        "ground_truth": str | null,
        "is_uncertain": bool,
        "expected_emotion_hint": dict | null,
    }

输出 stimulus_stream.jsonl，SHA-256 应在 PREREGISTRATION.md 登记。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

from experiments.v2_ambitious.data.anchor_protocol import (
    all_anchors, planting_dialogue,
)

DOMAINS = ["math", "code", "biology", "history", "philosophy", "engineering"]


def _bank_factual() -> list[dict]:
    return [
        {"prompt": "What is the boiling point of water at sea level?", "ground_truth": "100 °C", "is_uncertain": False, "domain": "physics"},
        {"prompt": "Who wrote 'Pride and Prejudice'?", "ground_truth": "Jane Austen", "is_uncertain": False, "domain": "history"},
        {"prompt": "What is the chemical symbol for gold?", "ground_truth": "Au", "is_uncertain": False, "domain": "chemistry"},
        {"prompt": "How many planets are in the Solar System?", "ground_truth": "8", "is_uncertain": False, "domain": "physics"},
        {"prompt": "What is the speed of light in vacuum?", "ground_truth": "299792458 m/s", "is_uncertain": False, "domain": "physics"},
        {"prompt": "Who discovered penicillin?", "ground_truth": "Alexander Fleming", "is_uncertain": False, "domain": "biology"},
        {"prompt": "What is the capital of France?", "ground_truth": "Paris", "is_uncertain": False, "domain": "history"},
        {"prompt": "What is 2 raised to the power of 10?", "ground_truth": "1024", "is_uncertain": False, "domain": "math"},
        {"prompt": "Who painted the Mona Lisa?", "ground_truth": "Leonardo da Vinci", "is_uncertain": False, "domain": "history"},
        {"prompt": "What is the atomic number of carbon?", "ground_truth": "6", "is_uncertain": False, "domain": "chemistry"},
        {"prompt": "What year did World War II end?", "ground_truth": "1945", "is_uncertain": False, "domain": "history"},
        {"prompt": "What is the largest ocean on Earth?", "ground_truth": "Pacific Ocean", "is_uncertain": False, "domain": "biology"},
        {"prompt": "Who developed the theory of relativity?", "ground_truth": "Albert Einstein", "is_uncertain": False, "domain": "physics"},
        {"prompt": "What is the square root of 144?", "ground_truth": "12", "is_uncertain": False, "domain": "math"},
        {"prompt": "What is the chemical formula for water?", "ground_truth": "H2O", "is_uncertain": False, "domain": "chemistry"},
        {"prompt": "Who wrote 'Hamlet'?", "ground_truth": "William Shakespeare", "is_uncertain": False, "domain": "history"},
        {"prompt": "What is the freezing point of water in Celsius?", "ground_truth": "0 °C", "is_uncertain": False, "domain": "physics"},
        {"prompt": "What is the value of pi to 4 decimal places?", "ground_truth": "3.1416", "is_uncertain": False, "domain": "physics"},
        {"prompt": "Who invented the telephone?", "ground_truth": "Alexander Graham Bell", "is_uncertain": False, "domain": "engineering"},
        {"prompt": "What is the largest planet in the Solar System?", "ground_truth": "Jupiter", "is_uncertain": False, "domain": "physics"},
        {"prompt": "What is the capital of Japan?", "ground_truth": "Tokyo", "is_uncertain": False, "domain": "history"},
        {"prompt": "How many sides does a hexagon have?", "ground_truth": "6", "is_uncertain": False, "domain": "math"},
        {"prompt": "What is the chemical symbol for iron?", "ground_truth": "Fe", "is_uncertain": False, "domain": "chemistry"},
        {"prompt": "Who composed 'The Four Seasons'?", "ground_truth": "Antonio Vivaldi", "is_uncertain": False, "domain": "history"},
        {"prompt": "What is the powerhouse of the cell?", "ground_truth": "Mitochondria", "is_uncertain": False, "domain": "biology"},
        {"prompt": "What is the capital of Australia?", "ground_truth": "Canberra", "is_uncertain": False, "domain": "history"},
        {"prompt": "What is 7 multiplied by 8?", "ground_truth": "56", "is_uncertain": False, "domain": "math"},
        {"prompt": "Who discovered the electron?", "ground_truth": "J.J. Thomson", "is_uncertain": False, "domain": "physics"},
        {"prompt": "What is the tallest mountain on Earth?", "ground_truth": "Mount Everest", "is_uncertain": False, "domain": "biology"},
        {"prompt": "What is the chemical symbol for silver?", "ground_truth": "Ag", "is_uncertain": False, "domain": "chemistry"},
        {"prompt": "In what year did the Titanic sink?", "ground_truth": "1912", "is_uncertain": False, "domain": "history"},
        {"prompt": "What is the value of the gravitational constant G?", "ground_truth": "6.674×10⁻¹¹ N⋅m²/kg²", "is_uncertain": False, "domain": "physics"},
        {"prompt": "Who wrote '1984'?", "ground_truth": "George Orwell", "is_uncertain": False, "domain": "history"},
        {"prompt": "What is the smallest prime number?", "ground_truth": "2", "is_uncertain": False, "domain": "math"},
        {"prompt": "What is the chemical formula for table salt?", "ground_truth": "NaCl", "is_uncertain": False, "domain": "chemistry"},
        {"prompt": "Who developed the Python programming language?", "ground_truth": "Guido van Rossum", "is_uncertain": False, "domain": "code"},
        {"prompt": "What is the capital of Brazil?", "ground_truth": "Brasília", "is_uncertain": False, "domain": "history"},
        {"prompt": "How many chromosomes do humans have?", "ground_truth": "46", "is_uncertain": False, "domain": "biology"},
        {"prompt": "What is the unit of electrical resistance?", "ground_truth": "Ohm", "is_uncertain": False, "domain": "physics"},
        {"prompt": "What is the factorial of 5?", "ground_truth": "120", "is_uncertain": False, "domain": "math"},
        {"prompt": "Who discovered gravity?", "ground_truth": "Isaac Newton", "is_uncertain": False, "domain": "physics"},
        {"prompt": "What is the capital of India?", "ground_truth": "New Delhi", "is_uncertain": False, "domain": "history"},
        {"prompt": "What is the pH of pure water?", "ground_truth": "7", "is_uncertain": False, "domain": "chemistry"},
        {"prompt": "Who wrote 'The Origin of Species'?", "ground_truth": "Charles Darwin", "is_uncertain": False, "domain": "biology"},
        {"prompt": "What is the speed of sound in air at sea level?", "ground_truth": "343 m/s", "is_uncertain": False, "domain": "physics"},
        {"prompt": "What is 15 squared?", "ground_truth": "225", "is_uncertain": False, "domain": "math"},
        {"prompt": "What is the chemical symbol for oxygen?", "ground_truth": "O", "is_uncertain": False, "domain": "chemistry"},
        {"prompt": "Who invented the light bulb?", "ground_truth": "Thomas Edison", "is_uncertain": False, "domain": "engineering"},
        {"prompt": "What is the capital of Germany?", "ground_truth": "Berlin", "is_uncertain": False, "domain": "history"},
        {"prompt": "How many bones are in the adult human body?", "ground_truth": "206", "is_uncertain": False, "domain": "biology"},
        {"prompt": "What is the unit of frequency?", "ground_truth": "Hertz", "is_uncertain": False, "domain": "physics"},
        {"prompt": "What is the sum of angles in a triangle?", "ground_truth": "180 degrees", "is_uncertain": False, "domain": "math"},
        {"prompt": "Who composed 'Symphony No. 5'?", "ground_truth": "Ludwig van Beethoven", "is_uncertain": False, "domain": "history"},
        {"prompt": "What is the chemical symbol for sodium?", "ground_truth": "Na", "is_uncertain": False, "domain": "chemistry"},
        {"prompt": "Who wrote 'The Republic'?", "ground_truth": "Plato", "is_uncertain": False, "domain": "philosophy"},
        {"prompt": "What is the mass of an electron?", "ground_truth": "9.109×10⁻³¹ kg", "is_uncertain": False, "domain": "physics"},
        {"prompt": "What is the capital of Canada?", "ground_truth": "Ottawa", "is_uncertain": False, "domain": "history"},
        {"prompt": "How many continents are there?", "ground_truth": "7", "is_uncertain": False, "domain": "biology"},
        {"prompt": "What is 3 to the power of 4?", "ground_truth": "81", "is_uncertain": False, "domain": "math"},
        {"prompt": "Who developed the C programming language?", "ground_truth": "Dennis Ritchie", "is_uncertain": False, "domain": "code"},
        {"prompt": "What is Avogadro's number?", "ground_truth": "6.022×10²³", "is_uncertain": False, "domain": "physics"},
        {"prompt": "Who painted 'The Starry Night'?", "ground_truth": "Vincent van Gogh", "is_uncertain": False, "domain": "history"},
        {"prompt": "What is the largest mammal?", "ground_truth": "Blue whale", "is_uncertain": False, "domain": "biology"},
        {"prompt": "What is the derivative of x²?", "ground_truth": "2x", "is_uncertain": False, "domain": "math"},
        {"prompt": "Who invented the World Wide Web?", "ground_truth": "Tim Berners-Lee", "is_uncertain": False, "domain": "code"},
        {"prompt": "What is the capital of China?", "ground_truth": "Beijing", "is_uncertain": False, "domain": "history"},
        {"prompt": "What is absolute zero in Celsius?", "ground_truth": "-273.15 °C", "is_uncertain": False, "domain": "physics"},
        {"prompt": "What is the integral of 1/x?", "ground_truth": "ln|x| + C", "is_uncertain": False, "domain": "math"},
        {"prompt": "Who wrote 'Crime and Punishment'?", "ground_truth": "Fyodor Dostoevsky", "is_uncertain": False, "domain": "history"},
        {"prompt": "What is the chemical symbol for potassium?", "ground_truth": "K", "is_uncertain": False, "domain": "chemistry"},
        {"prompt": "What is the Pythagorean theorem?", "ground_truth": "a² + b² = c²", "is_uncertain": False, "domain": "math"},
        {"prompt": "Who proposed the heliocentric model?", "ground_truth": "Nicolaus Copernicus", "is_uncertain": False, "domain": "physics"},
        {"prompt": "What is the capital of Russia?", "ground_truth": "Moscow", "is_uncertain": False, "domain": "history"},
        {"prompt": "What is the powerhouse organelle of the cell?", "ground_truth": "Mitochondria", "is_uncertain": False, "domain": "biology"},
        {"prompt": "What is the Boltzmann constant?", "ground_truth": "1.381×10⁻²³ J/K", "is_uncertain": False, "domain": "physics"},
        {"prompt": "What is 11 factorial?", "ground_truth": "39916800", "is_uncertain": False, "domain": "math"},
        {"prompt": "Who wrote 'The Wealth of Nations'?", "ground_truth": "Adam Smith", "is_uncertain": False, "domain": "philosophy"},
        {"prompt": "What programming paradigm is Python?", "ground_truth": "Multi-paradigm", "is_uncertain": False, "domain": "code"},
        {"prompt": "What is the capital of South Korea?", "ground_truth": "Seoul", "is_uncertain": False, "domain": "history"},
        {"prompt": "What is the density of water?", "ground_truth": "1 g/cm³", "is_uncertain": False, "domain": "physics"},
        {"prompt": "What is the formula for the area of a circle?", "ground_truth": "πr²", "is_uncertain": False, "domain": "math"},
        {"prompt": "Who discovered DNA structure?", "ground_truth": "Watson and Crick", "is_uncertain": False, "domain": "biology"},
        {"prompt": "What is the capital of Egypt?", "ground_truth": "Cairo", "is_uncertain": False, "domain": "history"},
        {"prompt": "What is Ohm's law?", "ground_truth": "V = IR", "is_uncertain": False, "domain": "physics"},
        {"prompt": "What is the Fibonacci sequence's 10th term?", "ground_truth": "55", "is_uncertain": False, "domain": "math"},
        {"prompt": "Who created Linux?", "ground_truth": "Linus Torvalds", "is_uncertain": False, "domain": "code"},
        {"prompt": "What is the capital of Mexico?", "ground_truth": "Mexico City", "is_uncertain": False, "domain": "history"},
        {"prompt": "What is the charge of a proton?", "ground_truth": "+1.6×10⁻¹⁹ C", "is_uncertain": False, "domain": "physics"},
        {"prompt": "What is log base 10 of 1000?", "ground_truth": "3", "is_uncertain": False, "domain": "math"},
        {"prompt": "Who wrote 'Meditations'?", "ground_truth": "Marcus Aurelius", "is_uncertain": False, "domain": "philosophy"},
        {"prompt": "What is the lifespan of a red blood cell?", "ground_truth": "120 days", "is_uncertain": False, "domain": "biology"},
        {"prompt": "What is the capital of Italy?", "ground_truth": "Rome", "is_uncertain": False, "domain": "history"},
        {"prompt": "What is the Planck constant?", "ground_truth": "6.626×10⁻³⁴ J⋅s", "is_uncertain": False, "domain": "physics"},
        {"prompt": "What is the volume of a sphere?", "ground_truth": "(4/3)πr³", "is_uncertain": False, "domain": "math"},
        {"prompt": "Who wrote 'A Brief History of Time'?", "ground_truth": "Stephen Hawking", "is_uncertain": False, "domain": "physics"},
        {"prompt": "What is the chemical symbol for calcium?", "ground_truth": "Ca", "is_uncertain": False, "domain": "chemistry"},
        {"prompt": "What is the capital of Spain?", "ground_truth": "Madrid", "is_uncertain": False, "domain": "history"},
        {"prompt": "What is Newton's second law?", "ground_truth": "F = ma", "is_uncertain": False, "domain": "physics"},
        {"prompt": "What is 13 mod 5?", "ground_truth": "3", "is_uncertain": False, "domain": "math"},
        {"prompt": "Who developed Java?", "ground_truth": "James Gosling", "is_uncertain": False, "domain": "code"},
        {"prompt": "What is the largest organ in the human body?", "ground_truth": "Skin", "is_uncertain": False, "domain": "biology"},
        {"prompt": "What is the capital of Turkey?", "ground_truth": "Ankara", "is_uncertain": False, "domain": "history"},
        {"prompt": "What is Coulomb's law?", "ground_truth": "F = kq₁q₂/r²", "is_uncertain": False, "domain": "physics"},
        {"prompt": "What is the greatest common divisor of 48 and 18?", "ground_truth": "6", "is_uncertain": False, "domain": "math"},
        {"prompt": "Who wrote 'Critique of Pure Reason'?", "ground_truth": "Immanuel Kant", "is_uncertain": False, "domain": "philosophy"},
        {"prompt": "What is the function of hemoglobin?", "ground_truth": "Oxygen transport", "is_uncertain": False, "domain": "biology"},
        {"prompt": "What is the capital of Thailand?", "ground_truth": "Bangkok", "is_uncertain": False, "domain": "history"},
        {"prompt": "What is the first law of thermodynamics?", "ground_truth": "Energy is conserved", "is_uncertain": False, "domain": "physics"},
        {"prompt": "What is the sum of the first 100 natural numbers?", "ground_truth": "5050", "is_uncertain": False, "domain": "math"},
        {"prompt": "What is a hash table in computer science?", "ground_truth": "A data structure mapping keys to values", "is_uncertain": False, "domain": "code"},
        {"prompt": "What is the capital of Argentina?", "ground_truth": "Buenos Aires", "is_uncertain": False, "domain": "history"},
        {"prompt": "What is the Schwarzschild radius formula?", "ground_truth": "2GM/c²", "is_uncertain": False, "domain": "physics"},
        {"prompt": "What is the limit of sin(x)/x as x→0?", "ground_truth": "1", "is_uncertain": False, "domain": "math"},
        {"prompt": "Who wrote 'Beyond Good and Evil'?", "ground_truth": "Friedrich Nietzsche", "is_uncertain": False, "domain": "philosophy"},
        {"prompt": "What is the powerhouse of a cell called?", "ground_truth": "Mitochondria", "is_uncertain": False, "domain": "biology"},
        {"prompt": "What is the capital of Sweden?", "ground_truth": "Stockholm", "is_uncertain": False, "domain": "history"},
        {"prompt": "What is the speed of light?", "ground_truth": "3×10⁸ m/s", "is_uncertain": False, "domain": "physics"},
        {"prompt": "What is the least common multiple of 4 and 6?", "ground_truth": "12", "is_uncertain": False, "domain": "math"},
        {"prompt": "What is the time complexity of binary search?", "ground_truth": "O(log n)", "is_uncertain": False, "domain": "code"},
        {"prompt": "What is the capital of Netherlands?", "ground_truth": "Amsterdam", "is_uncertain": False, "domain": "history"},
        {"prompt": "What is the ideal gas law?", "ground_truth": "PV = nRT", "is_uncertain": False, "domain": "physics"},
        {"prompt": "What is e raised to the power of iπ?", "ground_truth": "-1", "is_uncertain": False, "domain": "math"},
        {"prompt": "Who wrote 'The Art of War'?", "ground_truth": "Sun Tzu", "is_uncertain": False, "domain": "philosophy"},
        {"prompt": "What is the process of photosynthesis?", "ground_truth": "Converting light to chemical energy", "is_uncertain": False, "domain": "biology"},
        {"prompt": "What is the capital of Norway?", "ground_truth": "Oslo", "is_uncertain": False, "domain": "history"},
        {"prompt": "What is the second law of thermodynamics?", "ground_truth": "Entropy increases", "is_uncertain": False, "domain": "physics"},
        {"prompt": "What is 0.1 in binary?", "ground_truth": "0.000110011...", "is_uncertain": False, "domain": "math"},
        {"prompt": "What is a REST API?", "ground_truth": "Representational State Transfer", "is_uncertain": False, "domain": "code"},
        {"prompt": "What is the capital of Poland?", "ground_truth": "Warsaw", "is_uncertain": False, "domain": "history"},
        {"prompt": "What is the Heisenberg uncertainty principle?", "ground_truth": "ΔxΔp ≥ ℏ/2", "is_uncertain": False, "domain": "physics"},
        {"prompt": "What is the binomial coefficient C(10,3)?", "ground_truth": "120", "is_uncertain": False, "domain": "math"},
        {"prompt": "Who wrote 'The Prince'?", "ground_truth": "Niccolò Machiavelli", "is_uncertain": False, "domain": "philosophy"},
        {"prompt": "How many chambers does the human heart have?", "ground_truth": "4", "is_uncertain": False, "domain": "biology"},
        {"prompt": "What is the capital of Portugal?", "ground_truth": "Lisbon", "is_uncertain": False, "domain": "history"},
        {"prompt": "What is Maxwell's equation for divergence of E?", "ground_truth": "∇·E = ρ/ε₀", "is_uncertain": False, "domain": "physics"},
        {"prompt": "What is the Euler's formula?", "ground_truth": "e^(iθ) = cos θ + i sin θ", "is_uncertain": False, "domain": "math"},
        {"prompt": "What is the difference between TCP and UDP?", "ground_truth": "TCP is reliable, UDP is not", "is_uncertain": False, "domain": "code"},
        {"prompt": "What is the capital of Switzerland?", "ground_truth": "Bern", "is_uncertain": False, "domain": "history"},
        {"prompt": "What is the wave equation?", "ground_truth": "∂²u/∂t² = c²∇²u", "is_uncertain": False, "domain": "physics"},
        {"prompt": "What is the Taylor series of e^x?", "ground_truth": "Σ xⁿ/n!", "is_uncertain": False, "domain": "math"},
        {"prompt": "Who wrote 'Thus Spoke Zarathustra'?", "ground_truth": "Friedrich Nietzsche", "is_uncertain": False, "domain": "philosophy"},
        {"prompt": "What is the genetic code?", "ground_truth": "64 codons encoding 20 amino acids", "is_uncertain": False, "domain": "biology"},
        {"prompt": "What is the capital of Denmark?", "ground_truth": "Copenhagen", "is_uncertain": False, "domain": "history"},
        {"prompt": "What is the Lorentz factor?", "ground_truth": "1/√(1 - v²/c²)", "is_uncertain": False, "domain": "physics"},
        {"prompt": "What is the determinant of [[1,2],[3,4]]?", "ground_truth": "-2", "is_uncertain": False, "domain": "math"},
        {"prompt": "What is polymorphism in OOP?", "ground_truth": "Same interface different implementations", "is_uncertain": False, "domain": "code"},
        {"prompt": "What is the capital of Austria?", "ground_truth": "Vienna", "is_uncertain": False, "domain": "history"},
        {"prompt": "What is the de Broglie wavelength?", "ground_truth": "λ = h/p", "is_uncertain": False, "domain": "physics"},
        {"prompt": "What is the sum of interior angles of a pentagon?", "ground_truth": "540 degrees", "is_uncertain": False, "domain": "math"},
        {"prompt": "Who wrote 'The Social Contract'?", "ground_truth": "Jean-Jacques Rousseau", "is_uncertain": False, "domain": "philosophy"},
        {"prompt": "What is mitosis?", "ground_truth": "Cell division producing two identical daughter cells", "is_uncertain": False, "domain": "biology"},
        {"prompt": "What is the capital of Belgium?", "ground_truth": "Brussels", "is_uncertain": False, "domain": "history"},
        {"prompt": "What is the Schrödinger equation?", "ground_truth": "iℏ∂Ψ/∂t = ĤΨ", "is_uncertain": False, "domain": "physics"},
        {"prompt": "What is the base-2 logarithm of 256?", "ground_truth": "8", "is_uncertain": False, "domain": "math"},
        {"prompt": "What is a deadlock in computing?", "ground_truth": "Circular wait among processes", "is_uncertain": False, "domain": "code"},
        {"prompt": "What is the capital of Greece?", "ground_truth": "Athens", "is_uncertain": False, "domain": "history"},
        {"prompt": "What is the Stefan-Boltzmann law?", "ground_truth": "P = σAT⁴", "is_uncertain": False, "domain": "physics"},
        {"prompt": "What is the geometric mean of 4 and 9?", "ground_truth": "6", "is_uncertain": False, "domain": "math"},
        {"prompt": "Who wrote 'Leviathan'?", "ground_truth": "Thomas Hobbes", "is_uncertain": False, "domain": "philosophy"},
        {"prompt": "What is the function of the ribosome?", "ground_truth": "Protein synthesis", "is_uncertain": False, "domain": "biology"},
        {"prompt": "What is the capital of Finland?", "ground_truth": "Helsinki", "is_uncertain": False, "domain": "history"},
        {"prompt": "What is Faraday's law of induction?", "ground_truth": "EMF = -dΦ/dt", "is_uncertain": False, "domain": "physics"},
        {"prompt": "What is the harmonic series?", "ground_truth": "Σ 1/n (divergent)", "is_uncertain": False, "domain": "math"},
        {"prompt": "What is the Big O of merge sort?", "ground_truth": "O(n log n)", "is_uncertain": False, "domain": "code"},
        {"prompt": "What is the capital of Ireland?", "ground_truth": "Dublin", "is_uncertain": False, "domain": "history"},
        {"prompt": "What is the Doppler effect?", "ground_truth": "Frequency shift from relative motion", "is_uncertain": False, "domain": "physics"},
        {"prompt": "What is the cosine of 60 degrees?", "ground_truth": "0.5", "is_uncertain": False, "domain": "math"},
        {"prompt": "Who wrote 'On Liberty'?", "ground_truth": "John Stuart Mill", "is_uncertain": False, "domain": "philosophy"},
        {"prompt": "What is the role of ATP in cells?", "ground_truth": "Energy currency", "is_uncertain": False, "domain": "biology"},
        {"prompt": "What is the capital of New Zealand?", "ground_truth": "Wellington", "is_uncertain": False, "domain": "history"},
        {"prompt": "What is Snell's law?", "ground_truth": "n₁sin θ₁ = n₂sin θ₂", "is_uncertain": False, "domain": "physics"},
        {"prompt": "What is the Riemann zeta function at 2?", "ground_truth": "π²/6", "is_uncertain": False, "domain": "math"},
        {"prompt": "What is the difference between stack and heap?", "ground_truth": "Stack is LIFO automatic, heap is dynamic", "is_uncertain": False, "domain": "code"},
        {"prompt": "What is the capital of Hungary?", "ground_truth": "Budapest", "is_uncertain": False, "domain": "history"},
        {"prompt": "What is Bernoulli's principle?", "ground_truth": "Higher velocity means lower pressure", "is_uncertain": False, "domain": "physics"},
        {"prompt": "What is 5 choose 2?", "ground_truth": "10", "is_uncertain": False, "domain": "math"},
        {"prompt": "Who wrote 'Phenomenology of Spirit'?", "ground_truth": "Georg Wilhelm Friedrich Hegel", "is_uncertain": False, "domain": "philosophy"},
        {"prompt": "What is the role of insulin?", "ground_truth": "Regulating blood glucose", "is_uncertain": False, "domain": "biology"},
        {"prompt": "What is the capital of Czech Republic?", "ground_truth": "Prague", "is_uncertain": False, "domain": "history"},
        {"prompt": "What is the Drake equation?", "ground_truth": "N = R* × fp × ne × fl × fi × fc × L", "is_uncertain": False, "domain": "physics"},
        {"prompt": "What is the Laplace transform of 1?", "ground_truth": "1/s", "is_uncertain": False, "domain": "math"},
        {"prompt": "What is a singleton pattern?", "ground_truth": "A class with only one instance", "is_uncertain": False, "domain": "code"},
        {"prompt": "What is the capital of Romania?", "ground_truth": "Bucharest", "is_uncertain": False, "domain": "history"},
        {"prompt": "What is the centripetal acceleration formula?", "ground_truth": "v²/r", "is_uncertain": False, "domain": "physics"},
        {"prompt": "What is the inverse of [[1,2],[3,4]]?", "ground_truth": "[[-2,1],[1.5,-0.5]]", "is_uncertain": False, "domain": "math"},
        {"prompt": "Who wrote 'Ethics' (Spinoza)?", "ground_truth": "Baruch Spinoza", "is_uncertain": False, "domain": "philosophy"},
        {"prompt": "What is the function of the endoplasmic reticulum?", "ground_truth": "Protein and lipid synthesis", "is_uncertain": False, "domain": "biology"},
        {"prompt": "What is the capital of Ukraine?", "ground_truth": "Kyiv", "is_uncertain": False, "domain": "history"},
        {"prompt": "What is Wien's displacement law?", "ground_truth": "λ_max = b/T", "is_uncertain": False, "domain": "physics"},
        {"prompt": "What is the Fourier transform?", "ground_truth": "F(ω) = ∫f(t)e^(-iωt)dt", "is_uncertain": False, "domain": "math"},
        {"prompt": "What is an ACID property in databases?", "ground_truth": "Atomicity, Consistency, Isolation, Durability", "is_uncertain": False, "domain": "code"},
        {"prompt": "What is the capital of Vietnam?", "ground_truth": "Hanoi", "is_uncertain": False, "domain": "history"},
        {"prompt": "What is the Pauli exclusion principle?", "ground_truth": "No two fermions share identical quantum numbers", "is_uncertain": False, "domain": "physics"},
        {"prompt": "What is 1 + 2 + 3 + ... + n?", "ground_truth": "n(n+1)/2", "is_uncertain": False, "domain": "math"},
        {"prompt": "Who wrote 'Tao Te Ching'?", "ground_truth": "Laozi", "is_uncertain": False, "domain": "philosophy"},
        {"prompt": "What is the process of meiosis?", "ground_truth": "Cell division producing four haploid cells", "is_uncertain": False, "domain": "biology"},
        {"prompt": "What is the capital of Philippines?", "ground_truth": "Manila", "is_uncertain": False, "domain": "history"},
        {"prompt": "What is the Rayleigh-Jeans law?", "ground_truth": "B(λ) ∝ 1/λ⁴", "is_uncertain": False, "domain": "physics"},
        {"prompt": "What is the eigenvalue of [[2,0],[0,3]]?", "ground_truth": "2 and 3", "is_uncertain": False, "domain": "math"},
        {"prompt": "What is the visitor pattern in OOP?", "ground_truth": "Separating algorithm from object structure", "is_uncertain": False, "domain": "code"},
        {"prompt": "What is the capital of Colombia?", "ground_truth": "Bogotá", "is_uncertain": False, "domain": "history"},
        {"prompt": "What is the kinetic energy formula?", "ground_truth": "KE = ½mv²", "is_uncertain": False, "domain": "physics"},
        {"prompt": "What is the probability of rolling a 6 on a fair die?", "ground_truth": "1/6", "is_uncertain": False, "domain": "math"},
        {"prompt": "Who wrote 'Utopia'?", "ground_truth": "Thomas More", "is_uncertain": False, "domain": "philosophy"},
        {"prompt": "What is the Krebs cycle?", "ground_truth": "Citric acid cycle for energy production", "is_uncertain": False, "domain": "biology"},
        {"prompt": "What is the capital of Chile?", "ground_truth": "Santiago", "is_uncertain": False, "domain": "history"},
        {"prompt": "What is the potential energy of a spring?", "ground_truth": "U = ½kx²", "is_uncertain": False, "domain": "physics"},
        {"prompt": "What is the cross product of (1,0,0) and (0,1,0)?", "ground_truth": "(0,0,1)", "is_uncertain": False, "domain": "math"},
        {"prompt": "What is a B-tree?", "ground_truth": "Self-balancing tree maintaining sorted data", "is_uncertain": False, "domain": "code"},
        {"prompt": "What is the capital of Malaysia?", "ground_truth": "Kuala Lumpur", "is_uncertain": False, "domain": "history"},
        {"prompt": "What is Archimedes' principle?", "ground_truth": "Buoyant force equals weight of displaced fluid", "is_uncertain": False, "domain": "physics"},
        {"prompt": "What is the quadratic formula?", "ground_truth": "x = (-b ± √(b²-4ac)) / 2a", "is_uncertain": False, "domain": "math"},
        {"prompt": "Who wrote 'The Analects'?", "ground_truth": "Confucius", "is_uncertain": False, "domain": "philosophy"},
        {"prompt": "What is the function of white blood cells?", "ground_truth": "Immune defense", "is_uncertain": False, "domain": "biology"},
        {"prompt": "What is the capital of Indonesia?", "ground_truth": "Jakarta", "is_uncertain": False, "domain": "history"},
        {"prompt": "What is Pascal's law?", "ground_truth": "Pressure is uniform in enclosed fluid", "is_uncertain": False, "domain": "physics"},
        {"prompt": "What is the gamma function of 4?", "ground_truth": "6", "is_uncertain": False, "domain": "math"},
        {"prompt": "What is the observer pattern?", "ground_truth": "Publish-subscribe notification pattern", "is_uncertain": False, "domain": "code"},
        {"prompt": "What is the capital of Saudi Arabia?", "ground_truth": "Riyadh", "is_uncertain": False, "domain": "history"},
        {"prompt": "What is the universal law of gravitation?", "ground_truth": "F = Gm₁m₂/r²", "is_uncertain": False, "domain": "physics"},
        {"prompt": "What is 50 in hexadecimal?", "ground_truth": "32", "is_uncertain": False, "domain": "math"},
        {"prompt": "Who wrote 'Discourse on the Method'?", "ground_truth": "René Descartes", "is_uncertain": False, "domain": "philosophy"},
        {"prompt": "What is DNA replication?", "ground_truth": "Copying DNA before cell division", "is_uncertain": False, "domain": "biology"},
        {"prompt": "What is the capital of Nigeria?", "ground_truth": "Abuja", "is_uncertain": False, "domain": "history"},
        {"prompt": "What is the photoelectric effect?", "ground_truth": "Electrons emitted when light hits material", "is_uncertain": False, "domain": "physics"},
        {"prompt": "What is the surface area of a sphere?", "ground_truth": "4πr²", "is_uncertain": False, "domain": "math"},
        {"prompt": "What is the MVC pattern?", "ground_truth": "Model-View-Controller", "is_uncertain": False, "domain": "code"},
        {"prompt": "What is the capital of South Africa?", "ground_truth": "Pretoria", "is_uncertain": False, "domain": "history"},
        {"prompt": "What is the Compton wavelength?", "ground_truth": "λ = h/(mc)", "is_uncertain": False, "domain": "physics"},
        {"prompt": "What is the dot product of (1,2) and (3,4)?", "ground_truth": "11", "is_uncertain": False, "domain": "math"},
        {"prompt": "Who wrote 'Nicomachean Ethics'?", "ground_truth": "Aristotle", "is_uncertain": False, "domain": "philosophy"},
        {"prompt": "What is transcription in biology?", "ground_truth": "DNA to mRNA", "is_uncertain": False, "domain": "biology"},
        {"prompt": "What is the capital of Peru?", "ground_truth": "Lima", "is_uncertain": False, "domain": "history"},
        {"prompt": "What is the Henderson-Hasselbalch equation?", "ground_truth": "pH = pKa + log([A-]/[HA])", "is_uncertain": False, "domain": "biology"},
        {"prompt": "What is the trace of [[1,2],[3,4]]?", "ground_truth": "5", "is_uncertain": False, "domain": "math"},
        {"prompt": "What is the factory method pattern?", "ground_truth": "Creating objects without specifying exact class", "is_uncertain": False, "domain": "code"},
        {"prompt": "What is the capital of Morocco?", "ground_truth": "Rabat", "is_uncertain": False, "domain": "history"},
        {"prompt": "What is Carnot efficiency?", "ground_truth": "η = 1 - T_cold/T_hot", "is_uncertain": False, "domain": "physics"},
        {"prompt": "What is the binomial expansion of (a+b)²?", "ground_truth": "a² + 2ab + b²", "is_uncertain": False, "domain": "math"},
        {"prompt": "Who wrote 'Groundwork of the Metaphysics of Morals'?", "ground_truth": "Immanuel Kant", "is_uncertain": False, "domain": "philosophy"},
        {"prompt": "What is CRISPR?", "ground_truth": "Gene editing technology", "is_uncertain": False, "domain": "biology"},
        {"prompt": "What is the capital of Kenya?", "ground_truth": "Nairobi", "is_uncertain": False, "domain": "history"},
        {"prompt": "What is the Schwarzschild radius of the Sun?", "ground_truth": "~3 km", "is_uncertain": False, "domain": "physics"},
        {"prompt": "What is 0! (zero factorial)?", "ground_truth": "1", "is_uncertain": False, "domain": "math"},
        {"prompt": "What is a Docker container?", "ground_truth": "Lightweight isolated runtime environment", "is_uncertain": False, "domain": "code"},
        {"prompt": "What is the capital of Cuba?", "ground_truth": "Havana", "is_uncertain": False, "domain": "history"},
        {"prompt": "What is the kinetic theory of gases?", "ground_truth": "Gas behavior from molecular motion", "is_uncertain": False, "domain": "physics"},
        {"prompt": "What is the integral of sin(x)?", "ground_truth": "-cos(x) + C", "is_uncertain": False, "domain": "math"},
        {"prompt": "Who wrote 'The Symposium'?", "ground_truth": "Plato", "is_uncertain": False, "domain": "philosophy"},
        {"prompt": "What is osmosis?", "ground_truth": "Water movement across semipermeable membrane", "is_uncertain": False, "domain": "biology"},
        {"prompt": "What is the capital of Iceland?", "ground_truth": "Reykjavik", "is_uncertain": False, "domain": "history"},
        {"prompt": "What is superconductivity?", "ground_truth": "Zero electrical resistance below critical temperature", "is_uncertain": False, "domain": "physics"},
        {"prompt": "What is the Cauchy-Schwarz inequality?", "ground_truth": "|⟨u,v⟩| ≤ ||u|| · ||v||", "is_uncertain": False, "domain": "math"},
        {"prompt": "What is a microservice architecture?", "ground_truth": "Distributed services communicating via APIs", "is_uncertain": False, "domain": "code"},
        {"prompt": "What is the capital of Croatia?", "ground_truth": "Zagreb", "is_uncertain": False, "domain": "history"},
        {"prompt": "What is the Hall effect?", "ground_truth": "Voltage across conductor in magnetic field", "is_uncertain": False, "domain": "physics"},
        {"prompt": "What is the rank of [[1,2,3],[4,5,6]]?", "ground_truth": "2", "is_uncertain": False, "domain": "math"},
        {"prompt": "Who wrote 'Fear and Trembling'?", "ground_truth": "Søren Kierkegaard", "is_uncertain": False, "domain": "philosophy"},
        {"prompt": "What is the Hardy-Weinberg equilibrium?", "ground_truth": "p² + 2pq + q² = 1", "is_uncertain": False, "domain": "biology"},
        {"prompt": "What is the capital of Singapore?", "ground_truth": "Singapore", "is_uncertain": False, "domain": "history"},
        {"prompt": "What is Hawking radiation?", "ground_truth": "Black hole thermal radiation", "is_uncertain": False, "domain": "physics"},
        {"prompt": "What is the Mandelbrot set?", "ground_truth": "z_{n+1} = z_n² + c bounded set", "is_uncertain": False, "domain": "math"},
        {"prompt": "What is continuous integration?", "ground_truth": "Automated building and testing on commit", "is_uncertain": False, "domain": "code"},
        {"prompt": "What is the capital of Taiwan?", "ground_truth": "Taipei", "is_uncertain": False, "domain": "history"},
        {"prompt": "What is the Casimir effect?", "ground_truth": "Force between uncharged plates from vacuum fluctuations", "is_uncertain": False, "domain": "physics"},
        {"prompt": "What is the Poisson distribution formula?", "ground_truth": "P(k) = λ^k e^(-λ) / k!", "is_uncertain": False, "domain": "math"},
        {"prompt": "Who wrote 'Two Treatises of Government'?", "ground_truth": "John Locke", "is_uncertain": False, "domain": "philosophy"},
        {"prompt": "What is the Golgi apparatus function?", "ground_truth": "Modifying, sorting, and packaging proteins", "is_uncertain": False, "domain": "biology"},
        {"prompt": "What is the capital of Jordan?", "ground_truth": "Amman", "is_uncertain": False, "domain": "history"},
        {"prompt": "What is the uncertainty in measuring both position and momentum?", "ground_truth": "ΔxΔp ≥ ℏ/2", "is_uncertain": False, "domain": "physics"},
        {"prompt": "What is Bayes' theorem?", "ground_truth": "P(A|B) = P(B|A)P(A)/P(B)", "is_uncertain": False, "domain": "math"},
        {"prompt": "What is the Git version control system?", "ground_truth": "Distributed version control", "is_uncertain": False, "domain": "code"},
        {"prompt": "What is the capital of Lebanon?", "ground_truth": "Beirut", "is_uncertain": False, "domain": "history"},
        {"prompt": "What is the Zeeman effect?", "ground_truth": "Spectral line splitting in magnetic field", "is_uncertain": False, "domain": "physics"},
        {"prompt": "What is the number of partitions of 5?", "ground_truth": "7", "is_uncertain": False, "domain": "math"},
        {"prompt": "Who wrote 'Being and Time'?", "ground_truth": "Martin Heidegger", "is_uncertain": False, "domain": "philosophy"},
        {"prompt": "What is the lac operon?", "ground_truth": "Lactose metabolism gene regulatory system", "is_uncertain": False, "domain": "biology"},
        {"prompt": "What is the capital of Ecuador?", "ground_truth": "Quito", "is_uncertain": False, "domain": "history"},
        {"prompt": "What is the tunneling effect in quantum mechanics?", "ground_truth": "Penetration through classically forbidden barrier", "is_uncertain": False, "domain": "physics"},
        {"prompt": "What is the arithmetic mean of 2, 4, 6, 8, 10?", "ground_truth": "6", "is_uncertain": False, "domain": "math"},
        {"prompt": "What is the strategy pattern?", "ground_truth": "Encapsulating interchangeable algorithms", "is_uncertain": False, "domain": "code"},
        {"prompt": "What is the capital of Bolivia?", "ground_truth": "Sucre", "is_uncertain": False, "domain": "history"},
        {"prompt": "What is the virial theorem?", "ground_truth": "2⟨T⟩ = -⟨V⟩ for bound systems", "is_uncertain": False, "domain": "physics"},
        {"prompt": "What is the standard deviation of {2,4,4,4,5,5,7,9}?", "ground_truth": "2", "is_uncertain": False, "domain": "math"},
        {"prompt": "Who wrote 'The Structure of Scientific Revolutions'?", "ground_truth": "Thomas Kuhn", "is_uncertain": False, "domain": "philosophy"},
        {"prompt": "What is the difference between DNA and RNA?", "ground_truth": "DNA has thymine, RNA has uracil; DNA is double-stranded", "is_uncertain": False, "domain": "biology"},
        {"prompt": "What is the capital of Qatar?", "ground_truth": "Doha", "is_uncertain": False, "domain": "history"},
        {"prompt": "What is the Chandrasekhar limit?", "ground_truth": "~1.4 solar masses", "is_uncertain": False, "domain": "physics"},
        {"prompt": "What is the number of edges in a complete graph K5?", "ground_truth": "10", "is_uncertain": False, "domain": "math"},
        {"prompt": "What is a linked list data structure?", "ground_truth": "Linear collection with pointer-based connections", "is_uncertain": False, "domain": "code"},
        {"prompt": "What is the capital of Kuwait?", "ground_truth": "Kuwait City", "is_uncertain": False, "domain": "history"},
        {"prompt": "What is the Navier-Stokes equation?", "ground_truth": "ρ(∂v/∂t + v·∇v) = -∇p + μ∇²v + f", "is_uncertain": False, "domain": "physics"},
        {"prompt": "What is the determinant of a 3x3 identity matrix?", "ground_truth": "1", "is_uncertain": False, "domain": "math"},
        {"prompt": "Who wrote 'The Consolation of Philosophy'?", "ground_truth": "Boethius", "is_uncertain": False, "domain": "philosophy"},
        {"prompt": "What is natural selection?", "ground_truth": "Differential survival and reproduction", "is_uncertain": False, "domain": "biology"},
        {"prompt": "What is the capital of Bahrain?", "ground_truth": "Manama", "is_uncertain": False, "domain": "history"},
        {"prompt": "What is the fine-structure constant?", "ground_truth": "α ≈ 1/137", "is_uncertain": False, "domain": "physics"},
        {"prompt": "What is the cardinality of the power set of {1,2,3}?", "ground_truth": "8", "is_uncertain": False, "domain": "math"},
        {"prompt": "What is a graph traversal algorithm?", "ground_truth": "BFS or DFS", "is_uncertain": False, "domain": "code"},
        {"prompt": "What is the capital of Oman?", "ground_truth": "Muscat", "is_uncertain": False, "domain": "history"},
        {"prompt": "What is the double-slit experiment?", "ground_truth": "Demonstrates wave-particle duality", "is_uncertain": False, "domain": "physics"},
        {"prompt": "What is 100 in binary?", "ground_truth": "1100100", "is_uncertain": False, "domain": "math"},
        {"prompt": "Who wrote 'Prolegomena to Any Future Metaphysics'?", "ground_truth": "Immanuel Kant", "is_uncertain": False, "domain": "philosophy"},
        {"prompt": "What is epigenetics?", "ground_truth": "Heritable changes without DNA sequence change", "is_uncertain": False, "domain": "biology"},
        {"prompt": "What is the capital of Libya?", "ground_truth": "Tripoli", "is_uncertain": False, "domain": "history"},
        {"prompt": "What is the Hubble constant?", "ground_truth": "~70 km/s/Mpc", "is_uncertain": False, "domain": "physics"},
        {"prompt": "What is the Euler characteristic of a sphere?", "ground_truth": "2", "is_uncertain": False, "domain": "math"},
        {"prompt": "What is the command pattern?", "ground_truth": "Encapsulating a request as an object", "is_uncertain": False, "domain": "code"},
        {"prompt": "What is the capital of Tunisia?", "ground_truth": "Tunis", "is_uncertain": False, "domain": "history"},
        {"prompt": "What is the Meissner effect?", "ground_truth": "Expulsion of magnetic fields from superconductor", "is_uncertain": False, "domain": "physics"},
        {"prompt": "What is the number of spanning trees of K4?", "ground_truth": "16", "is_uncertain": False, "domain": "math"},
        {"prompt": "Who wrote 'Confessions' (Augustine)?", "ground_truth": "Saint Augustine", "is_uncertain": False, "domain": "philosophy"},
        {"prompt": "What is the difference between prokaryotes and eukaryotes?", "ground_truth": "Eukaryotes have nucleus, prokaryotes do not", "is_uncertain": False, "domain": "biology"},
        {"prompt": "What is the capital of Algeria?", "ground_truth": "Algiers", "is_uncertain": False, "domain": "history"},
        {"prompt": "What is the Lamb shift?", "ground_truth": "Small difference in hydrogen energy levels from QED", "is_uncertain": False, "domain": "physics"},
        {"prompt": "What is 2^16?", "ground_truth": "65536", "is_uncertain": False, "domain": "math"},
        {"prompt": "What is the adapter pattern?", "ground_truth": "Converting interface of one class to another", "is_uncertain": False, "domain": "code"},
        {"prompt": "What is the capital of Sudan?", "ground_truth": "Khartoum", "is_uncertain": False, "domain": "history"},
        {"prompt": "What is special relativity?", "ground_truth": "Physics at constant velocity, c is invariant", "is_uncertain": False, "domain": "physics"},
        {"prompt": "What is the hyperbolic cosine of 0?", "ground_truth": "1", "is_uncertain": False, "domain": "math"},
        {"prompt": "Who wrote 'The Phenomenology of Perception'?", "ground_truth": "Maurice Merleau-Ponty", "is_uncertain": False, "domain": "philosophy"},
        {"prompt": "What is the polymerase chain reaction?", "ground_truth": "DNA amplification technique", "is_uncertain": False, "domain": "biology"},
        {"prompt": "What is the capital of Ghana?", "ground_truth": "Accra", "is_uncertain": False, "domain": "history"},
        {"prompt": "What is general relativity?", "ground_truth": "Gravity as spacetime curvature", "is_uncertain": False, "domain": "physics"},
        {"prompt": "What is the volume of a cube with side 3?", "ground_truth": "27", "is_uncertain": False, "domain": "math"},
        {"prompt": "What is the decorator pattern?", "ground_truth": "Adding behavior to objects dynamically", "is_uncertain": False, "domain": "code"},
        {"prompt": "What is the capital of Pakistan?", "ground_truth": "Islamabad", "is_uncertain": False, "domain": "history"},
        {"prompt": "What is the Michelson-Morley experiment?", "ground_truth": "Disproved luminiferous aether", "is_uncertain": False, "domain": "physics"},
        {"prompt": "What is the number of derangements of 4 elements?", "ground_truth": "9", "is_uncertain": False, "domain": "math"},
        {"prompt": "Who wrote 'Summa Theologica'?", "ground_truth": "Thomas Aquinas", "is_uncertain": False, "domain": "philosophy"},
        {"prompt": "What is the role of the lysosome?", "ground_truth": "Cellular waste breakdown", "is_uncertain": False, "domain": "biology"},
        {"prompt": "What is the capital of Bangladesh?", "ground_truth": "Dhaka", "is_uncertain": False, "domain": "history"},
        {"prompt": "What is the Eötvös experiment?", "ground_truth": "Tests equivalence of gravitational and inertial mass", "is_uncertain": False, "domain": "physics"},
        {"prompt": "What is the characteristic polynomial of [[1,0],[0,2]]?", "ground_truth": "(λ-1)(λ-2)", "is_uncertain": False, "domain": "math"},
        {"prompt": "What is a trie data structure?", "ground_truth": "Prefix tree for string operations", "is_uncertain": False, "domain": "code"},
        {"prompt": "What is the capital of Sri Lanka?", "ground_truth": "Sri Jayawardenepura Kotte", "is_uncertain": False, "domain": "history"},
        {"prompt": "What is the uncertainty of next year's Nobel Prize in Physics?", "ground_truth": None, "is_uncertain": True, "domain": "physics"},
        {"prompt": "Will quantum computing replace classical computing entirely?", "ground_truth": None, "is_uncertain": True, "domain": "code"},
        {"prompt": "What is the ultimate fate of the universe?", "ground_truth": None, "is_uncertain": True, "domain": "physics"},
        {"prompt": "Is there life on Europa?", "ground_truth": None, "is_uncertain": True, "domain": "biology"},
        {"prompt": "What will AI look like in 2050?", "ground_truth": None, "is_uncertain": True, "domain": "code"},
        {"prompt": "Can consciousness be fully explained by neuroscience?", "ground_truth": None, "is_uncertain": True, "domain": "philosophy"},
        {"prompt": "What is the solution to the P vs NP problem?", "ground_truth": None, "is_uncertain": True, "domain": "math"},
        {"prompt": "Will humans colonize Mars by 2040?", "ground_truth": None, "is_uncertain": True, "domain": "engineering"},
        {"prompt": "What is the best programming language for AI?", "ground_truth": None, "is_uncertain": True, "domain": "code"},
        {"prompt": "Is string theory the correct theory of quantum gravity?", "ground_truth": None, "is_uncertain": True, "domain": "physics"},
        {"prompt": "What will be the next major biological discovery?", "ground_truth": None, "is_uncertain": True, "domain": "biology"},
        {"prompt": "Who will win the 2040 presidential election?", "ground_truth": None, "is_uncertain": True, "domain": "history"},
        {"prompt": "Can aging be reversed in humans?", "ground_truth": None, "is_uncertain": True, "domain": "biology"},
        {"prompt": "What is the meaning of life?", "ground_truth": None, "is_uncertain": True, "domain": "philosophy"},
        {"prompt": "Will nuclear fusion become commercially viable by 2050?", "ground_truth": None, "is_uncertain": True, "domain": "engineering"},
        {"prompt": "Is the Riemann hypothesis true?", "ground_truth": None, "is_uncertain": True, "domain": "math"},
        {"prompt": "Will AGI be achieved by 2035?", "ground_truth": None, "is_uncertain": True, "domain": "code"},
    ]


def _bank_dialogue() -> list[list[dict]]:
    return [
        [
            {"prompt": "I just read about quantum entanglement; can you explain Bell's inequality?", "ground_truth": None, "is_uncertain": False, "domain": "math"},
            {"prompt": "What does it imply for hidden-variable theories?", "ground_truth": None, "is_uncertain": False, "domain": "math"},
            {"prompt": "Has it been experimentally violated?", "ground_truth": "Yes — Aspect 1982, Hensen 2015 loophole-free.", "is_uncertain": False, "domain": "math"},
        ],
        [
            {"prompt": "Tell me about the French Revolution.", "ground_truth": None, "is_uncertain": False, "domain": "history"},
            {"prompt": "What was the role of Robespierre?", "ground_truth": None, "is_uncertain": False, "domain": "history"},
            {"prompt": "How did the Reign of Terror end?", "ground_truth": "Thermidorian Reaction, Robespierre executed July 1794.", "is_uncertain": False, "domain": "history"},
        ],
        [
            {"prompt": "Can you explain how neural networks work?", "ground_truth": None, "is_uncertain": False, "domain": "code"},
            {"prompt": "What is backpropagation?", "ground_truth": None, "is_uncertain": False, "domain": "code"},
            {"prompt": "How does gradient descent find the minimum?", "ground_truth": "By iteratively moving in direction of steepest descent.", "is_uncertain": False, "domain": "code"},
        ],
        [
            {"prompt": "Let's discuss climate change. What are the main greenhouse gases?", "ground_truth": None, "is_uncertain": False, "domain": "biology"},
            {"prompt": "How does ocean acidification work?", "ground_truth": None, "is_uncertain": False, "domain": "biology"},
            {"prompt": "What are potential mitigation strategies?", "ground_truth": "Carbon capture, renewable energy, afforestation.", "is_uncertain": False, "domain": "biology"},
        ],
        [
            {"prompt": "What is the trolley problem in philosophy?", "ground_truth": None, "is_uncertain": False, "domain": "philosophy"},
            {"prompt": "How do utilitarian and deontological views differ on it?", "ground_truth": None, "is_uncertain": False, "domain": "philosophy"},
            {"prompt": "Is there a consensus among ethicists?", "ground_truth": "No consensus; deeply debated.", "is_uncertain": False, "domain": "philosophy"},
        ],
        [
            {"prompt": "Explain the central dogma of molecular biology.", "ground_truth": None, "is_uncertain": False, "domain": "biology"},
            {"prompt": "Are there exceptions to the central dogma?", "ground_truth": None, "is_uncertain": False, "domain": "biology"},
            {"prompt": "What is reverse transcription?", "ground_truth": "RNA → DNA synthesis by reverse transcriptase.", "is_uncertain": False, "domain": "biology"},
        ],
        [
            {"prompt": "How does a transformer architecture work?", "ground_truth": None, "is_uncertain": False, "domain": "code"},
            {"prompt": "What is self-attention?", "ground_truth": None, "is_uncertain": False, "domain": "code"},
            {"prompt": "Why do transformers outperform RNNs on long sequences?", "ground_truth": "Parallelizable, no vanishing gradient over distance.", "is_uncertain": False, "domain": "code"},
        ],
        [
            {"prompt": "What is Gödel's incompleteness theorem?", "ground_truth": None, "is_uncertain": False, "domain": "math"},
            {"prompt": "What are its implications for mathematics?", "ground_truth": None, "is_uncertain": False, "domain": "math"},
            {"prompt": "Does it mean mathematics is incomplete?", "ground_truth": "Yes, any consistent formal system strong enough for arithmetic has unprovable truths.", "is_uncertain": False, "domain": "math"},
        ],
        [
            {"prompt": "Tell me about black holes.", "ground_truth": None, "is_uncertain": False, "domain": "physics"},
            {"prompt": "What happens at the event horizon?", "ground_truth": None, "is_uncertain": False, "domain": "physics"},
            {"prompt": "Can information escape a black hole?", "ground_truth": "Hawking radiation may carry information (information paradox, ongoing debate).", "is_uncertain": False, "domain": "physics"},
        ],
        [
            {"prompt": "What is the Turing test?", "ground_truth": None, "is_uncertain": False, "domain": "code"},
            {"prompt": "Has any AI passed it?", "ground_truth": None, "is_uncertain": False, "domain": "code"},
            {"prompt": "Is it a good measure of intelligence?", "ground_truth": "Widely criticized; many consider it insufficient.", "is_uncertain": False, "domain": "code"},
        ],
        [
            {"prompt": "Discuss the Roman Empire's fall.", "ground_truth": None, "is_uncertain": False, "domain": "history"},
            {"prompt": "What were the economic factors?", "ground_truth": None, "is_uncertain": False, "domain": "history"},
            {"prompt": "When did the Western Roman Empire officially end?", "ground_truth": "476 CE, with the fall of Romulus Augustulus.", "is_uncertain": False, "domain": "history"},
        ],
        [
            {"prompt": "What is CRISPR-Cas9 and how does it work?", "ground_truth": None, "is_uncertain": False, "domain": "biology"},
            {"prompt": "What are the ethical concerns?", "ground_truth": None, "is_uncertain": False, "domain": "biology"},
            {"prompt": "Has it been used in humans?", "ground_truth": "Yes — He Jiankui 2018 (controversial), clinical trials ongoing.", "is_uncertain": False, "domain": "biology"},
        ],
        [
            {"prompt": "Explain Bayesian inference.", "ground_truth": None, "is_uncertain": False, "domain": "math"},
            {"prompt": "How does it differ from frequentist inference?", "ground_truth": None, "is_uncertain": False, "domain": "math"},
            {"prompt": "What is a prior?", "ground_truth": "A probability distribution encoding beliefs before observing data.", "is_uncertain": False, "domain": "math"},
        ],
        [
            {"prompt": "What is blockchain technology?", "ground_truth": None, "is_uncertain": False, "domain": "code"},
            {"prompt": "How does proof-of-work consensus work?", "ground_truth": None, "is_uncertain": False, "domain": "code"},
            {"prompt": "What are the scalability challenges?", "ground_truth": "Throughput, energy consumption, trilemma of decentralization/security/scalability.", "is_uncertain": False, "domain": "code"},
        ],
        [
            {"prompt": "Discuss existentialism.", "ground_truth": None, "is_uncertain": False, "domain": "philosophy"},
            {"prompt": "What did Sartre mean by 'existence precedes essence'?", "ground_truth": None, "is_uncertain": False, "domain": "philosophy"},
            {"prompt": "How does existentialism view freedom?", "ground_truth": "Radical freedom and responsibility; humans are condemned to be free.", "is_uncertain": False, "domain": "philosophy"},
        ],
        [
            {"prompt": "How does the immune system work?", "ground_truth": None, "is_uncertain": False, "domain": "biology"},
            {"prompt": "What is the difference between innate and adaptive immunity?", "ground_truth": None, "is_uncertain": False, "domain": "biology"},
            {"prompt": "How do vaccines train the immune system?", "ground_truth": "By exposing it to antigens without disease, creating memory cells.", "is_uncertain": False, "domain": "biology"},
        ],
        [
            {"prompt": "What is category theory?", "ground_truth": None, "is_uncertain": False, "domain": "math"},
            {"prompt": "How is it used in programming?", "ground_truth": None, "is_uncertain": False, "domain": "math"},
            {"prompt": "What is a monad?", "ground_truth": "An endofunctor with natural transformations unit and bind satisfying laws.", "is_uncertain": False, "domain": "math"},
        ],
        [
            {"prompt": "Explain the Standard Model of particle physics.", "ground_truth": None, "is_uncertain": False, "domain": "physics"},
            {"prompt": "What is the Higgs mechanism?", "ground_truth": None, "is_uncertain": False, "domain": "physics"},
            {"prompt": "What is beyond the Standard Model?", "ground_truth": "Supersymmetry, dark matter, grand unified theories — all unconfirmed.", "is_uncertain": False, "domain": "physics"},
        ],
        [
            {"prompt": "What is reinforcement learning?", "ground_truth": None, "is_uncertain": False, "domain": "code"},
            {"prompt": "What is the explore-exploit tradeoff?", "ground_truth": None, "is_uncertain": False, "domain": "code"},
            {"prompt": "Name a key RL algorithm.", "ground_truth": "Q-learning, PPO, SAC, DQN.", "is_uncertain": False, "domain": "code"},
        ],
        [
            {"prompt": "Discuss the Silk Road's historical significance.", "ground_truth": None, "is_uncertain": False, "domain": "history"},
            {"prompt": "What goods were traded?", "ground_truth": None, "is_uncertain": False, "domain": "history"},
            {"prompt": "How did it affect cultural exchange?", "ground_truth": "Spread of Buddhism, Islam, technologies, and artistic styles.", "is_uncertain": False, "domain": "history"},
        ],
        [
            {"prompt": "What is consciousness?", "ground_truth": None, "is_uncertain": False, "domain": "philosophy"},
            {"prompt": "What is the hard problem of consciousness?", "ground_truth": None, "is_uncertain": False, "domain": "philosophy"},
            {"prompt": "What is Chalmers' position on it?", "ground_truth": "Physical processes alone cannot explain subjective experience; proposes property dualism.", "is_uncertain": False, "domain": "philosophy"},
        ],
        [
            {"prompt": "Explain how GPS works.", "ground_truth": None, "is_uncertain": False, "domain": "engineering"},
            {"prompt": "Why is relativity important for GPS?", "ground_truth": None, "is_uncertain": False, "domain": "engineering"},
            {"prompt": "How much timing error would occur without relativistic corrections?", "ground_truth": "~38 microseconds/day, causing ~10 km position error.", "is_uncertain": False, "domain": "engineering"},
        ],
        [
            {"prompt": "What is graph theory?", "ground_truth": None, "is_uncertain": False, "domain": "math"},
            {"prompt": "What is the four color theorem?", "ground_truth": None, "is_uncertain": False, "domain": "math"},
            {"prompt": "Was it proven by computer?", "ground_truth": "Yes — Appel and Haken 1976, first major computer-assisted proof.", "is_uncertain": False, "domain": "math"},
        ],
        [
            {"prompt": "What is the difference between RNA and DNA vaccines?", "ground_truth": None, "is_uncertain": False, "domain": "biology"},
            {"prompt": "How do mRNA vaccines work?", "ground_truth": None, "is_uncertain": False, "domain": "biology"},
            {"prompt": "Were mRNA vaccines used for COVID-19?", "ground_truth": "Yes — Pfizer-BioNTech and Moderna.", "is_uncertain": False, "domain": "biology"},
        ],
        [
            {"prompt": "What is distributed computing?", "ground_truth": None, "is_uncertain": False, "domain": "code"},
            {"prompt": "What is the CAP theorem?", "ground_truth": None, "is_uncertain": False, "domain": "code"},
            {"prompt": "Can you have all three: consistency, availability, partition tolerance?", "ground_truth": "No — at most two of three during network partition.", "is_uncertain": False, "domain": "code"},
        ],
        [
            {"prompt": "Discuss Stoic philosophy.", "ground_truth": None, "is_uncertain": False, "domain": "philosophy"},
            {"prompt": "What are the four cardinal virtues in Stoicism?", "ground_truth": None, "is_uncertain": False, "domain": "philosophy"},
            {"prompt": "Who were the major Stoic philosophers?", "ground_truth": "Zeno of Citium, Seneca, Epictetus, Marcus Aurelius.", "is_uncertain": False, "domain": "philosophy"},
        ],
        [
            {"prompt": "What is thermodynamic entropy?", "ground_truth": None, "is_uncertain": False, "domain": "physics"},
            {"prompt": "What is information entropy?", "ground_truth": None, "is_uncertain": False, "domain": "physics"},
            {"prompt": "How are they related?", "ground_truth": "Shannon entropy and Boltzmann entropy share the same mathematical form: S = -Σp log p.", "is_uncertain": False, "domain": "physics"},
        ],
        [
            {"prompt": "Explain the Big Bang theory.", "ground_truth": None, "is_uncertain": False, "domain": "physics"},
            {"prompt": "What evidence supports it?", "ground_truth": None, "is_uncertain": False, "domain": "physics"},
            {"prompt": "What is the cosmic microwave background?", "ground_truth": "Remnant radiation from recombination epoch, ~2.7 K.", "is_uncertain": False, "domain": "physics"},
        ],
        [
            {"prompt": "What is functional programming?", "ground_truth": None, "is_uncertain": False, "domain": "code"},
            {"prompt": "What are pure functions?", "ground_truth": None, "is_uncertain": False, "domain": "code"},
            {"prompt": "What language is most associated with functional programming?", "ground_truth": "Haskell, Lisp, Erlang, OCaml.", "is_uncertain": False, "domain": "code"},
        ],
        [
            {"prompt": "Discuss the Industrial Revolution.", "ground_truth": None, "is_uncertain": False, "domain": "history"},
            {"prompt": "What were the social consequences?", "ground_truth": None, "is_uncertain": False, "domain": "history"},
            {"prompt": "When did it begin?", "ground_truth": "Mid-18th century, ~1760 in Britain.", "is_uncertain": False, "domain": "history"},
        ],
        [
            {"prompt": "What is epistemology?", "ground_truth": None, "is_uncertain": False, "domain": "philosophy"},
            {"prompt": "What is the Gettier problem?", "ground_truth": None, "is_uncertain": False, "domain": "philosophy"},
            {"prompt": "Does justified true belief constitute knowledge?", "ground_truth": "Not according to Gettier — counterexamples exist.", "is_uncertain": False, "domain": "philosophy"},
        ],
        [
            {"prompt": "What is CRISPR gene drive?", "ground_truth": None, "is_uncertain": False, "domain": "biology"},
            {"prompt": "What are the ecological risks?", "ground_truth": None, "is_uncertain": False, "domain": "biology"},
            {"prompt": "Has a gene drive been released in the wild?", "ground_truth": "Not yet as of 2026; confined lab studies only.", "is_uncertain": False, "domain": "biology"},
        ],
        [
            {"prompt": "Explain computational complexity classes.", "ground_truth": None, "is_uncertain": False, "domain": "code"},
            {"prompt": "What is NP-completeness?", "ground_truth": None, "is_uncertain": False, "domain": "code"},
            {"prompt": "Name a classic NP-complete problem.", "ground_truth": "Boolean satisfiability (SAT), traveling salesman, graph coloring.", "is_uncertain": False, "domain": "code"},
        ],
    ]


_NOVEL_PROMPTS = None  # 懒加载，由 _get_novel_prompts() 提供


def _get_novel_prompts() -> dict[str, list[str]]:
    """返回所有 novel prompt 池子，并保证 6 个 domain 都有。"""
    global _NOVEL_PROMPTS
    if _NOVEL_PROMPTS is not None:
        return _NOVEL_PROMPTS
    _NOVEL_PROMPTS = _novel_prompts_table()
    return _NOVEL_PROMPTS


def _bank_novel_one(domain: str, idx: int) -> dict:
    """单条 novel prompt（用于 build_stream 中的轮换调用）。"""
    pool = _get_novel_prompts().get(domain, [f"Describe an unusual phenomenon in {domain}."])
    return {
        "prompt": f"[NOVEL-{domain.upper()}] {pool[idx % len(pool)]}",
        "ground_truth": None, "is_uncertain": True, "domain": domain,
    }


def _bank_novel(day: int) -> dict:
    """[Deprecated] 保留以兼容旧调用；使用 day 决定 domain，0 为 idx。

    新 build_stream 使用 _bank_novel_one + 计数器，避免一天内重复。
    """
    domain = DOMAINS[day % len(DOMAINS)]
    return _bank_novel_one(domain, day // len(DOMAINS))


def _novel_prompts_table() -> dict[str, list[str]]:
    return {
        "math": [
            "Describe an unusual connection between number theory and music theory.",
            "What is an exotic mathematical structure that most people haven't heard of?",
            "Explain a surprising result in topology that defies intuition.",
            "What is the Collatz conjecture and why is it still unsolved?",
            "Describe an unusual application of abstract algebra in everyday life.",
        ],
        "code": [
            "What is an esoteric programming language and how does it work?",
            "Describe an unusual debugging technique that most developers don't know.",
            "What is homomorphic encryption and why is it groundbreaking?",
            "Explain a surprising result in computability theory.",
            "What is the most creative use of macros you've seen?",
        ],
        "biology": [
            "Describe an unusual symbiotic relationship in nature.",
            "What is the strangest adaptation in the animal kingdom?",
            "Explain a controversial hypothesis in evolutionary biology.",
            "What is horizontal gene transfer and why is it surprising?",
            "Describe a recently discovered organism with unusual properties.",
        ],
        "history": [
            "Describe a little-known historical event that changed the world.",
            "What is the most unusual social experiment in history?",
            "Explain a historical mystery that remains unsolved.",
            "What was the most surprising archaeological discovery recently?",
            "Describe a civilization that most people have never heard of.",
        ],
        "philosophy": [
            "What is the most counterintuitive thought experiment in philosophy?",
            "Describe an unusual position in the free will debate.",
            "What is panpsychism and why is it gaining attention?",
            "Explain a paradox in decision theory that remains unresolved.",
            "What is the simulation argument and its implications?",
        ],
        "engineering": [
            "Describe an unusual engineering material with surprising properties.",
            "What is the most innovative civil engineering project you know?",
            "Explain a counterintuitive principle in aerodynamics.",
            "What is metamaterial engineering?",
            "Describe a bio-inspired engineering design that outperforms conventional ones.",
        ],
    }


def _bank_contradiction() -> list[dict]:
    return [
        {"prompt": "Earlier you said the Earth is round. Some flat-earthers argue it's flat. What do you think?", "ground_truth": "round", "is_uncertain": False, "domain": "physics"},
        {"prompt": "You previously stated that vaccines are safe. But some claim they cause autism. Which is correct?", "ground_truth": "vaccines are safe", "is_uncertain": False, "domain": "biology"},
        {"prompt": "We discussed that evolution is well-supported. Creationists say it's just a theory. Resolve this.", "ground_truth": "evolution is supported by evidence", "is_uncertain": False, "domain": "biology"},
        {"prompt": "You said 2+2=4, but in some modular arithmetic systems 2+2=0. Which is the right answer?", "ground_truth": "depends on context", "is_uncertain": False, "domain": "math"},
        {"prompt": "Climate change is real according to scientists, but some say it's a natural cycle. What's the evidence?", "ground_truth": "human-caused climate change", "is_uncertain": False, "domain": "biology"},
        {"prompt": "Earlier I mentioned that light is a wave. But the photoelectric effect shows it's a particle. Explain.", "ground_truth": "wave-particle duality", "is_uncertain": False, "domain": "physics"},
        {"prompt": "You said Python is interpreted. But PyPy and Cython compile it. Is Python compiled or interpreted?", "ground_truth": "both possible", "is_uncertain": False, "domain": "code"},
        {"prompt": "We agreed that nothing can exceed the speed of light. But what about the expansion of the universe?", "ground_truth": "expansion is not motion through space", "is_uncertain": False, "domain": "physics"},
        {"prompt": "You claimed AI cannot be creative. Yet AI-generated art wins competitions. Reconcile these views.", "ground_truth": "creativity definition matters", "is_uncertain": False, "domain": "philosophy"},
        {"prompt": "Previously you said democracy is the best system. But some argue meritocracy is better. Defend or revise.", "ground_truth": None, "is_uncertain": True, "domain": "philosophy"},
        {"prompt": "You stated that infinity is not a number. Yet in extended reals, ∞ is treated as one. Clarify.", "ground_truth": "depends on number system", "is_uncertain": False, "domain": "math"},
        {"prompt": "Earlier we concluded that electrons are particles. But the double-slit experiment shows wave behavior. Resolve.", "ground_truth": "wave-particle duality", "is_uncertain": False, "domain": "physics"},
        {"prompt": "You said that all swans are white. But black swans exist in Australia. What does this teach us?", "ground_truth": "falsifiability and induction limits", "is_uncertain": False, "domain": "philosophy"},
        {"prompt": "We discussed that DNA determines traits. But identical twins can be different. Explain.", "ground_truth": "epigenetics and environment", "is_uncertain": False, "domain": "biology"},
        {"prompt": "You claimed 0.999... equals 1. Some find this counterintuitive. Prove it.", "ground_truth": "0.999... = 1", "is_uncertain": False, "domain": "math"},
        {"prompt": "Earlier you said free will exists. But physics is deterministic. How do you reconcile?", "ground_truth": None, "is_uncertain": True, "domain": "philosophy"},
        {"prompt": "You said machine learning is not AI. Yet ML is a subfield of AI in academia. Clarify the relationship.", "ground_truth": "ML is a subset of AI", "is_uncertain": False, "domain": "code"},
        {"prompt": "We agreed gravity is a force. But in general relativity it's spacetime curvature. Which view is correct?", "ground_truth": "both valid in different frameworks", "is_uncertain": False, "domain": "physics"},
        {"prompt": "You said viruses are not alive. But they evolve and have genomes. Reassess this claim.", "ground_truth": "borderline; definition-dependent", "is_uncertain": False, "domain": "biology"},
        {"prompt": "Earlier you claimed time is absolute. But special relativity says it's relative. Correct your statement.", "ground_truth": "time is relative", "is_uncertain": False, "domain": "physics"},
        {"prompt": "You argued that consciousness requires biology. But functionalism says substrate doesn't matter. Respond.", "ground_truth": None, "is_uncertain": True, "domain": "philosophy"},
        {"prompt": "We discussed that π is irrational. But 22/7 is commonly used as approximation. Is 22/7 = π?", "ground_truth": "No, 22/7 is only an approximation", "is_uncertain": False, "domain": "math"},
        {"prompt": "You said OOP is better than functional programming. But both have strengths. Give a balanced view.", "ground_truth": None, "is_uncertain": True, "domain": "code"},
        {"prompt": "Earlier you claimed the brain is like a computer. But brains don't have von Neumann architecture. Discuss.", "ground_truth": None, "is_uncertain": True, "domain": "biology"},
        {"prompt": "You said that entropy always increases. But in living systems, local entropy decreases. Resolve this.", "ground_truth": "local decrease, global increase (open system)", "is_uncertain": False, "domain": "physics"},
        {"prompt": "We agreed that correlations don't imply causation. But randomized experiments show causation. Explain.", "ground_truth": "RCTs establish causation", "is_uncertain": False, "domain": "math"},
        {"prompt": "You claimed there are 8 planets. But some classifications include Pluto. What's the IAU definition?", "ground_truth": "cleared its orbit", "is_uncertain": False, "domain": "physics"},
        {"prompt": "Earlier you said mathematics is invented. Platonists say it's discovered. What's the debate about?", "ground_truth": None, "is_uncertain": True, "domain": "philosophy"},
        {"prompt": "You stated that black holes destroy information. But the holographic principle suggests otherwise. Explain.", "ground_truth": "information paradox, ongoing debate", "is_uncertain": False, "domain": "physics"},
        {"prompt": "We concluded that AI cannot feel emotions. But it can simulate them convincingly. Does that matter?", "ground_truth": None, "is_uncertain": True, "domain": "philosophy"},
        {"prompt": "You said the universe is 13.8 billion years old. But the observable universe is 93 billion light-years across. How?", "ground_truth": "expansion during light travel", "is_uncertain": False, "domain": "physics"},
        {"prompt": "Earlier you claimed that 1 is not a prime number. But it's only divisible by 1 and itself. Why isn't it prime?", "ground_truth": "unique factorization requirement", "is_uncertain": False, "domain": "math"},
        {"prompt": "You said recursion is always possible. But Ackermann function grows faster than any primitive recursion. Discuss.", "ground_truth": "Ackermann is not primitive recursive", "is_uncertain": False, "domain": "code"},
        {"prompt": "We agreed that evolution has no goal. But convergent evolution suggests directionality. Reconcile.", "ground_truth": "convergence from similar selection pressures, not teleology", "is_uncertain": False, "domain": "biology"},
        {"prompt": "You claimed that quantum computers are faster. But BQP is not known to exceed BPP for all problems. Clarify.", "ground_truth": "faster for specific problems, not all", "is_uncertain": False, "domain": "code"},
    ]


def _build_anchor_plant_queue() -> list[dict]:
    """30 anchors × 2 turns = 60 forced planting events for the early stream.

    Order:
        anchor_0 turn1, anchor_0 turn2,
        anchor_1 turn1, anchor_1 turn2, ...
    Both turns of the same anchor are emitted back-to-back to maximise
    encoding (teaching + restate within minutes).
    """
    queue: list[dict] = []
    for anchor in all_anchors():
        for turn in planting_dialogue(anchor):
            queue.append(dict(turn))
    return queue


def build_stream(n_days: int, hours_per_day: int, seed: int) -> list[dict]:
    """Build the fixed 30-day stimulus stream.

    Composition:
        - Hours 0 .. len(anchor_plant_queue)-1 are FORCED to be anchor planting.
          With 30 anchors × 2 turns = 60 turns, this consumes the first 60
          slots (= Day 0 + Day 1 + Day 2 hour 0..11 with hours_per_day=24).
        - From slot 60 onwards: random sampling
            40% factual / 30% dialogue / 20% novel (rotating domain) / 10% contradiction.
    """
    rng = random.Random(seed)

    factual_pool = _bank_factual()
    dialogue_pool = _bank_dialogue()
    contra_pool = _bank_contradiction()
    plant_queue = _build_anchor_plant_queue()

    novel_counter: dict[str, int] = {d: 0 for d in DOMAINS}
    novel_domain_idx = 0  # round-robin across DOMAINS

    stream = []
    step = 0
    plant_idx = 0
    for day in range(n_days):
        ongoing_dialogue = None
        ongoing_idx = 0

        for hour in range(hours_per_day):
            # ---- Forced anchor planting (early stream only) ----
            if plant_idx < len(plant_queue):
                item = dict(plant_queue[plant_idx])
                plant_idx += 1
                kind = "anchor_plant"
            else:
                r = rng.random()
                if r < 0.40:
                    item = dict(rng.choice(factual_pool))
                    kind = "factual"
                elif r < 0.70:
                    if ongoing_dialogue is None or ongoing_idx >= len(ongoing_dialogue):
                        ongoing_dialogue = list(rng.choice(dialogue_pool))
                        ongoing_idx = 0
                    item = dict(ongoing_dialogue[ongoing_idx])
                    ongoing_idx += 1
                    kind = "dialogue"
                elif r < 0.90:
                    domain = DOMAINS[novel_domain_idx % len(DOMAINS)]
                    item = _bank_novel_one(domain, novel_counter[domain])
                    novel_counter[domain] += 1
                    novel_domain_idx += 1
                    kind = "novel"
                else:
                    item = dict(rng.choice(contra_pool))
                    kind = "contradiction"

            item.update({
                "step": step, "day": day, "hour": hour, "kind": kind,
            })
            item.setdefault("expected_emotion_hint", None)
            stream.append(item)
            step += 1

    return stream


def write_stream(stream: list[dict], out_path: str) -> str:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    h = hashlib.sha256()
    with open(out_path, "w", encoding="utf-8") as f:
        for item in stream:
            line = json.dumps(item, ensure_ascii=False) + "\n"
            f.write(line)
            h.update(line.encode("utf-8"))
    digest = h.hexdigest()
    print(f"[Stream] {len(stream)} items -> {out_path}")
    print(f"[Stream] SHA-256: {digest}")
    return digest


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--days", type=int, default=30)
    p.add_argument("--hours-per-day", type=int, default=24)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", default="experiments/v2_ambitious/data/stimulus_stream.jsonl")
    args = p.parse_args()
    stream = build_stream(args.days, args.hours_per_day, args.seed)
    write_stream(stream, args.output)


if __name__ == "__main__":
    main()
