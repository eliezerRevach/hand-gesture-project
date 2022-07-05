using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.Threading;
namespace Autohand
{
    public class handCC : MonoBehaviour
    {
        public Hand autoHand;
        public pos pos;

        public void Start()
        {
           timer();
        }
        //fingers a float values bettwen 0 to 1 , by setting values 0 to 1 to each finger in bendOffset , can be created each gesture wanted
        public void setPaper()
        {
            autoHand.fingers[0].bendOffset = 0;
            autoHand.fingers[1].bendOffset = 0;
            autoHand.fingers[2].bendOffset = 0;
            autoHand.fingers[3].bendOffset = 0;
            autoHand.fingers[4].bendOffset = 0;

        }


        public void setRock()
        {
            autoHand.fingers[0].bendOffset = 1;
            autoHand.fingers[1].bendOffset = 1;
            autoHand.fingers[2].bendOffset = 1;
            autoHand.fingers[3].bendOffset = 1;
            autoHand.fingers[4].bendOffset = 1;
        }
        public void setscissors()
        {

            autoHand.fingers[0].bendOffset = 0;
            autoHand.fingers[1].bendOffset = 0;
            autoHand.fingers[2].bendOffset = 1;
            autoHand.fingers[3].bendOffset = 1;
            autoHand.fingers[4].bendOffset = 1;


        }

        public void timer()
        {

            Debug.Log("starting");
            StartCoroutine(inputWaiter());
        }

         public IEnumerator inputWaiter()
         {

          
             while (true){
                 if (main.gesture == "rock")
                 {
                     setRock();

                 }

                 if (main.gesture == "paper")
                 {
                     setPaper();
                 }
                 

                 if (main.gesture == "scissors")
                 {
                     setscissors();
                 }

                 yield return new WaitForSeconds(1);
             }
         }

    }
}
