import * as React from 'react';
import { Text, View, StyleSheet } from 'react-native';
import Constants from 'expo-constants';
import * as Firebase from 'firebase';
import { Card } from 'react-native-paper';
export default class SearchScreen extends React.Component { 
constructor()
{
super();
this.state = {
  TextVal : "",
}
}
render() {
const {navigation,route} = this.props;
const { Object1 } = route.params;
var o = Object1;

const firebaseConfig = {
  apiKey: "AIzaSyAf0zJlQT3EHqpUnUKLlc_JwS_BR6zlWSg",
  authDomain: "tarp-alzheimer.firebaseapp.com",
  databaseURL: "https://tarp-alzheimer.firebaseio.com",
  projectId: "tarp-alzheimer",
  storageBucket: "tarp-alzheimer.appspot.com",
  messagingSenderId: "615755110137",
  appId: "1:615755110137:web:bb04e83d2597a1c5b3f428",
  measurementId: "G-KD1NXG9FE1"
};
var s;
var database;
if(Firebase.app.length == 0)
{
  Firebase.initializeApp(firebaseConfig);
}
database = Firebase.database();

var ref = database.ref('TestData');
ref.on('value',gotData,errData)

function gotData(data)
{
  var o1 = o;
  var scores = data.val();
  var keys = Object.keys(scores);
  //console.log(keys);
  s = '';
  console.log("o1:" + o1);
  for(var i = 0;i<keys.length;i++)
  {
    var flag = 0;
    var s1 = s;
    var k = keys[i];
    var k3 = 1;
    var arr = [scores[k].data1,scores[k].data2,scores[k].data3,scores[k].data4,scores[k].data5,scores[k].data6,scores[k].data7,scores[k].data8,scores[k].data9,scores[k].data10,scores[k].data11,scores[k].data12,scores[k].data13,scores[k].data14,scores[k].data15];
    console.log(scores[k].length);
    s = '';
    for(var i1 = 0;i1<arr.length;i1++)
    {
      if(arr[i1] == o1)
      {
        s = s + '';
        flag = 1;
      }
      else if(arr[i1] != "")
      {
        s = s + k3 + "." + arr[i1]+ "  ";
        k3 = k3 + 1;
      }
    }
    console.log(s + "  " + i);
    if(flag == 0)
    {
       s = s1;
    }
  }
  console.log(s);
} 
const o2 = s;
function errData(err)
{
  console.log('Error');
  console.log(err) ;
}
  return(
    <View style = {styles.container}>
    <Text>You can find the required object near</Text>
    <Text>{JSON.stringify(o2)}</Text>
    </View>
  );
}
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: 'center',
    paddingTop: Constants.statusBarHeight,
    backgroundColor: '#0AE',
    padding: 8,
    fontSize : 40,
  },
  paragraph: {
    margin: 24,
    fontSize: 18,
    fontWeight: 'bold',
    textAlign: 'center',
  },
});
