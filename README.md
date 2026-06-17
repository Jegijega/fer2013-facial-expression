# სახის ემოციების ამოცნობა (FER2013)

ეს არის ჩემი ამოხსნა Kaggle-ის შეჯიბრისთვის *Challenges in Representation
Learning: Facial Expression Recognition Challenge*. ამოცანა ისაა, რომ 48x48
ზომის შავ-თეთრი სახის ფოტოს მიხედვით ვიწინასწარმეტყველო 7 ემოცია:
ბრაზი, ზიზღი, შიში, სიხარული, სევდა, გაკვირვება, ნეიტრალური.
ყველა ექსპერიმენტი ცალკე run-ად ჩავწერე
Weights & Biases-ზე, რომ ერთმანეთს შევადარო.

## შედეგები

**linear** (ლოგისტიკური რეგრესია პიქსელებზე, ~16K პარამეტრი): test acc 0.364,
train-val სხვაობა ~0. underfit-ია, სივრცული სტრუქტურის წვდომა არ აქვს.

**tiny_cnn** (2 conv ბლოკი, ~0.6M): test acc 0.526, სხვაობა ~0.47. conv ბევრს
შველის, მაგრამ რეგულარიზაციის გარეშე უკვე overfit-ს აკეთებს.

**deeper_cnn** (4 conv ბლოკი, რეგულარიზაციის გარეშე, ~5.3M): test acc 0.584,
სხვაობა ~0.42. ძლიერად overfit-ს აკეთებს, იზეპირებს სავარჯიშო მონაცემებს.

**regularized_cnn** (იგივე სიღრმე + BN + dropout + augmentation, ~3.5M): test acc
0.704, სხვაობა ~0.03. ჩემი საუკეთესო from-scratch მოდელი.

**resnet18** (transfer learning, ImageNet, ~11M): test acc 0.716, სხვაობა ~0.28.
ყველაზე მაღალი accuracy, მაგრამ უფრო მეტად overfit-ს აკეთებს.

## როგორ არის რეპო დალაგებული

```
README.md ეს ფაილი
requirements.txt
train.py ერთი config = ერთი W&B run
configs/ თითო ექსპერიმენტზე თითო YAML
src/
  data.py FER2013-ის წაკითხვა, official splits, augmentation
  models.py ყველა მოდელი + registry, რომ სახელით ავაგო
  engine.py training / eval ციკლი და მთელი W&B დალოგვა
  sanity.py forward/backward შემოწმებები
  utils.py seed-ები, მეტრიკები, confusion matrix-ის დახატვა
scripts/
  download_data.sh მონაცემების გადმოწერა Kaggle API-ით
  make_notebook.py Colab notebook-ის თავიდან გენერაცია
notebooks/
  fer2013_colab.ipynb notebook, რომელსაც Colab-ზე ვუშვებ
reports/
  EXPERIMENTS.md დეტალური ანალიზი თითო run-ზე
  REPORT.md ჩემი ჩანაწერები W&B report-ისთვის
```

## მონაცემები

dataset ერთი CSV-ია სამი სვეტით: emotion, pixels (2304 პიქსელი, ანუ 48x48,
ერთმანეთისგან გამოყოფილი) და Usage. ვიყენებ ორიგინალ split-ებს: Training
სავარჯიშოდ, PublicTest ვალიდაციისთვის და PrivateTest საბოლოო ტესტისთვის.
ეს split-ები შენარჩუნდა განზრახ, რომ ჩემი ტესტის რიცხვები ძველ leaderboard-ს
შევადარო.

მონაცემები ნორმალიზებულია dataset-ის mean/std-ით. აქ ერთი მნიშვნელოვანი რამ
კლასების დისბალანსია: "Happy" ბევრია და "Disgust" მთელი
მონაცემების მხოლოდ ~1.5%-ია. სწორედ ამიტომ ვლოგავ per-class accuracy-სა და
confusion matrix-ს თითო run-ზე, და არა მხოლოდ საერთო რიცხვს.

## ჩემი ექსპერიმენტები, თანმიმდევრობით

მოდელები ნაბიჯ-ნაბიჯ ავაგე. თითო ნაბიჯი გადაწყვეტილებაა და თითო ცალკე W&B
run-ია.

**0. Linear baseline.** მხოლოდ ერთი linear ფენა ბრტყელ პიქსელებზე. მას მხოლოდ
პიქსელების გლობალური აწონვა შეუძლია, არა ნიმუშების ცნობა, ამიტომ ~33%-ზე ჩერდება.
train და val accuracy ორივე დაბალია და ერთმანეთთან ახლოს. ეს underfitting-ის
კლასიკური ნიშანია. მე მას ქვედა საზომად ვიყენებ: რაც ამას ვერ აჯობებს, ის არ ვარგა.

**1. TinyCNN.** ორი convolution ბლოკი. ეს პირველი მოდელია, რომელიც ნამდვილად
ლოკალურ ნიმუშებს უყურებს, და accuracy ~50%-მდე ადის. ამან დამიდასტურა რომ
convolution სწორი იარაღია. მაგრამ საინტერესო ისაა, რომ ეს "პატარა" CNN-იც კი
overfit-ს იწყებს, train acc 98%-მდე ადის, val კი ~52%-ზე იყინება. ეს უბიძგებს
ჯერ უფრო დიდი მოდელისკენ (overfit რომ უარესი გავხადო) და მერე რეგულარიზაციისკენ.

**2. DeeperCNN (overfit-ის დემო).** ოთხი conv ბლოკი და დიდი FC თავი, მაგრამ
განზრახ BatchNorm-ის, dropout-ის, augmentation-ისა და weight decay-ის გარეშე.
ამდენი capacity-თ და ნულოვანი რეგულარიზაციით train accuracy ~99%-ს უახლოვდება,
ხოლო val mid-50-ებში ჩერდება, სხვაობა უზარმაზარია (`generalization_gap`-ს
ვლოგავ ყოველ epoch-ზე). val loss ზევით მიდის მაშინ, როცა train loss ეცემა.
ეს run არის ანალიზისთვის.

**3. RegularizedCNN.** იგივე სიღრმე, მაგრამ ახლა რეგულარიზატორებს თითო-თითოდ
ვამატებ: BatchNorm (უფრო სტაბილური ვარჯიში), Dropout (ფენების co-adaptation-ის
თავიდან ასაცილებლად) და data augmentation (random flip-ები და მცირე
rotation/shift-ები, რაც სახეებისთვის ყველაზე მეტს შველის). ამას ემატება weight
decay, ცოტა label smoothing და cosine learning-rate schedule. სხვაობა მკვეთრად
მცირდება და test accuracy ~70%-მდე ადის. ეს ჩემი საუკეთესო from-scratch
მოდელია.

**4. ResNet18 (transfer learning).** მინდოდა მენახა, ცნობილი არქიტექტურა
აჯობებდა თუ არა ჩემს ხელით აშენებულს. torchvision-ის ResNet18-ს ვიღებ, პირველ
conv-ს ვცვლი 1 არხზე და ვაშორებ ადრეულ downsampling-ს (48x48 სახე პატარაა,
რეზოლუციის თავიდანვე დაკარგვა არ მინდა), და ვტვირთავ ImageNet-ის წონებს. უფრო
სწრაფად ისწავლა და ყველაზე მაღალი accuracy მიიღო, ~72%. მაგრამ 11M პარამეტრის
გამო augmentation-ის მიუხედავად მაინც overfit-ს იწყებს მეორე ნახევარში - ამიტომ
ეს ყველაზე ზუსტი მოდელია, მაგრამ არა ყველაზე "სუფთა".

## Sanity checks

სანამ რომელიმე მრუდს დავუჯერებ, ვუშვებ შემოწმებებს src/sanity.py-დან
(notebook-ის მე-5 ნაწილი):

1. Forward: output-ის ფორმაა (B, 7) და ახალი მოდელის პირველი loss ≈ ln(7) =
   1.946. თუ ასე არ არის, logits/loss/ინიციალიზაცია არასწორია.
2. Gradients: ერთი backward pass ყველა პარამეტრს აძლევს რეალურ, სასრულ
   გრადიენტს (არც მკვდარი შტო, არც გაწყვეტილი გრაფი).
3. ერთი batch-ის overfit: მოდელმა ერთი პატარა batch ~100%-მდე უნდა "დაიზეპიროს".
   თუ 16 სურათს ვერ იზეპირებს, 28K-ს ვერასდროს ისწავლის, ანუ ბაგი კოდშია, არა
   მონაცემებში.

ამ შემოწმებებს ხუთივე მოდელზე ვუშვებ, რომ დრო არ დავკარგო არასწორ მოდელზე.

## რას ვლოგავ W&B-ზე

ეს ისე მოვაწყვე, რომ წინა დავალების MLflow-ის სტრუქტურას დაემსგავსოს: თითო
მოდელზე თითო run, და run-ები დაჯგუფებულია baselines, from_scratch და
transfer_learning-ად.

- ყველა ჰიპერპარამეტრი მიდის wandb.config-ში (მოდელი, learning rate, batch
  size, optimizer, scheduler, weight decay, dropout, augmentation და ა.შ.) +
  პარამეტრების რაოდენობა.
- მეტრიკები ყოველ epoch-ზე: train/val loss და accuracy, learning rate და
  generalization gap.
- wandb.watch ლოგავს გრადიენტებისა და წონების ჰისტოგრამებს, რაც გამოსადეგია
  vanishing/exploding gradient-ების შესამჩნევად.
- summary მნიშვნელობები: საუკეთესო val accuracy, test accuracy, per-class test
  accuracy, პარამეტრების რაოდენობა.
- confusion matrix ტესტ მონაცემებზე.

## ჰიპერპარამეტრების გადარჩევა

ჰიპერპარამეტრები თითო მოდელზე ხელით გადავარჩიე (ჩანს configs-ში და თითო run-ის
W&B config-ში): სხვადასხვა learning rate (1e-3 და 5e-4), optimizer (adam და
adamw), dropout, weight decay და cosine schedule უფრო ღრმა მოდელებისთვის.
ყველაზე დიდი გავლენა learning rate-მა და რეგულარიზაციამ (augmentation + dropout)
მოახდინა.

## რა გამოვიტანე ამ ყველაფრიდან

- მარტო capacity პასუხი არ არის. ღრმა, არარეგულარიზებული მოდელი ვალიდაციაზე
  ძლივს ჯობნის პატარას, მიუხედავად იმისა, რომ სავარჯიშო მონაცემები დაიზეპირა.
- ამ dataset-ისთვის augmentation ყველაზე სასარგებლო რეგულარიზატორი აღმოჩნდა.
- იშვიათი კლასები ბოლომდე რთული რჩება. ჩემი საუკეთესო მოდელიც კი სუსტია
  "Disgust"-ზე და ერევა fear/sad/neutral ერთმანეთში, რაც ნათლად ჩანს confusion
  matrix-ში.
- მთლიანი გზა underfit -> overfit -> რეგულარიზებული -> transfer სწორედ ის
  bias/variance ისტორიაა, რომელიც ლექციაზე გავიარეთ, ოღონდ ახლა რეალურ მრუდებზე.

##  wandb link: 
  https://wandb.ai/gabas22-free-university-of-tbilisi-/fer2013-fer-challenge/overview