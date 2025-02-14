# The MIT License

# Copyright (c) 2025 Albert Murienne

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import context

from gallek_translator import gallek, kellag
from  tools import apertium

fr_text = "traduis de français en breton: j'apprends le breton à l'école."

gk = gallek()
result = gk.translate_fr2br(fr_text)

print("input fr text: " + fr_text)
print("Gallek translation: " + result)
print("Apertium reverse translation: " + apertium.translate_br2fr(result))

br_text = "treiñ eus ar galleg d'ar brezhoneg : deskiñ a ran brezhoneg er skol."

kg = kellag()
result = kg.translate_br2fr(br_text)

print("input br text: " + br_text)
print("Kellag translation: " + result)

