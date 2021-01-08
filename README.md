# <div align="center">Retrieval-Chatbot (ChatterBot)</div>

<BR><BR><BR>

- ChatterBot is a Casual Conversational Chatbot which is developed following the Retrieval Based Approach of making Chatbots & Natural Language Processing.Here what we have implemented is a Chatbot which will mimic
  human like casual conversation.

- For this we have used our own intents.json file which contains various intents/situations/scenarios.

- Each of the intent has an identifier tag and a list of probable user inputs to that scenario along with a list of responses for the same.

- Each of these input sentences is passed to KERAS Tokenizer, where all the tokenized form of these sentences is returned.This is essential so that the sentences can be further passed to the Embedding layer which returns an Embedding matrix which is eventually passed to the Dense Layers.

- A Deep Learning Model is employed for its working where in it is trained on this intents so as to correctly identify the "tag" of any incoming user text after which just a randomly selected repsonse from the list of responses is returned to the user.

- Flask API is used to deploy this python project into a web application with a beautiful User Interface to give the exact experience of what a person would feel while having a conversation with a real person.


<BR>

## 📸 Screenshots

<table>
  <tr>
    <th>Screenshot 1</th>
    <th>Screenshot 2</th>
  </tr>
  
  <tr>
    <td><img src="https://github.com/adwaitgondhalekar/Retrieval-Chatbot/blob/master/Screenshots/Screenshot1.png"></td>
    <td><img src="https://github.com/adwaitgondhalekar/Retrieval-Chatbot/blob/master/Screenshots/Screenshot2.png"></td>
  </tr>
  
  
</table>


<BR>

## Libraries used  🛠

- Keras
- Tensorflow
- Scikit-Learn
- Pandas

<BR>

## 📱 Creators  🤝

  With the current levels of globalization it's necessary to be able to communicate one's ideas to the opposite person in an understandable way. <BR> So with that we are working in a team of 4 towards our Mini Project.
- [Atharva Kulkarni](https://kulkarniatharva.github.io)
- [Adwait Gondhalekar](https://github.com/adwaitgondhalekar)
- [Radha Mujumdar](https://github.com/radhamujumdar)
- [Shreya Kedia](https://github.com/shreya-kedia)
