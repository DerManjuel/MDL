# MDL Fragen

## Exercise 1
1. Warum kann man CrossEntropy Loss verwenden? --> die einzelnen Pixel werden klassifiziert
2. Irgendwas zu den Shortcuts?

## Exercise 2
1. Identity mapping - was ist das und wofür verwendet man es?
    Rechte Seite = feature learning --> Gefahr von vanishing gradient, da viele layer und wenig infos, keine Lernfortschritte in Backpropagation (wenn in Kettenregel ein Glied = 0, ist die ganze Kette 0)
    --> bei Resnets funktioniert Netz immer noch, wenn einzelne Teile nicht lernen
    Linke Seite = identity mapping --> leicht modifizierter input wird wieder draufaddiert, um den Gradienten wieder zu erhöhen und vanishing gradient entgegenzuwirken

identity mapping = skip connections mit wenig layern

2. Was passiert im LSTM? Was machen die einzelnen Gates? - input, forget, output, state
    - forget gate: sigmoid funktion: entweder alles wird vergessen oder alles wird behalten vom vorherigen Cell state --> Gewichtung ergibt sich aus dem vorherigen hidden state + neuem Input
    - input gate: welche neuen Infos werden behalten und zum cell state addiert (bestimmt sigmoid, welcher Teil behalten wird und tanh die Gewichtung)
    - output gate: was kommt in den neuen hidden state
    - state gate: was ist der cell state output für den nächsten layer --> kombi aus was wird vergessen, was wird behalten?

## Exercise 3
1. Wie funktionieren Joint Histograms und was sagen sie aus?
2. Was passiert in dieser Zeile 
ssd[idx] = torch.sum((feat_patch_fixed - feat_displacements)**2).squeeze() 
3. Warum gibt es mehrere Branches? 
