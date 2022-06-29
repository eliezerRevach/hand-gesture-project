// using System;
// using Unity.VisualScripting;
// using UnityEngine;
//
// using System.Collections;
// namespace Autohand
// {
//
//     public class handCC : MonoBehaviour
//     {
//         
//         public static Hand autoHand;
//         public static pos pos;
//
//         public void Start()
//         { 
//             StartCoroutine(inputWaiter());
//         }
//
//         public  void setPaper()
//         {
//
//             autoHand.fingers[0].bendOffset = 0;
//             autoHand.fingers[1].bendOffset = 0;
//             autoHand.fingers[2].bendOffset = 0;
//             autoHand.fingers[3].bendOffset = 0;
//             autoHand.fingers[4].bendOffset = 0;
//
//         }
//
//         public  void setRock()
//         {
//             Debug.Log("rock");
//             autoHand.fingers[0].bendOffset = 1;
//             autoHand.fingers[1].bendOffset = 1;
//             autoHand.fingers[2].bendOffset = 1;
//             autoHand.fingers[3].bendOffset = 1;
//             autoHand.fingers[4].bendOffset = 1;
//         }
//         public  void setscissors()
//         {
//
//             autoHand.fingers[0].bendOffset = 1;
//             autoHand.fingers[1].bendOffset = 0;
//             autoHand.fingers[2].bendOffset = 1;
//             autoHand.fingers[3].bendOffset = 0;
//             autoHand.fingers[4].bendOffset = 1;
//
//
//         }
//         
//         // ReSharper disable Unity.PerformanceAnalysis
//         public IEnumerator inputWaiter()
//         {
//             while (true){
//                 if (main.gesture == "rock")
//                 {
//                     Debug.Log("rock");
//                     setRock();
//
//                 }
//
//                 if (main.gesture == "paper")
//                 {
//                     Debug.Log("paper");
//                     setPaper();
//                 }
//                 
//
//                 if (main.gesture == "scissors")
//                 {
//                     Debug.Log("scissors");
//                     setscissors();
//                 }
//
//                 yield return new WaitForSeconds(1);
//             }
//         }
//     }
//     
// }


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

        public void setPaper()
        {
            // pos.setPose(new Vector3(0, 0, 1), new Vector3(0, 0, 0));
            autoHand.fingers[0].bendOffset = 0;
            autoHand.fingers[1].bendOffset = 0;
            autoHand.fingers[2].bendOffset = 0;
            autoHand.fingers[3].bendOffset = 0;
            autoHand.fingers[4].bendOffset = 0;

        }


        public void setRock()
        {
            // pos.setPose(new Vector3(0, 0, 2), new Vector3(0, 0, 0));
            autoHand.fingers[0].bendOffset = 1;
            autoHand.fingers[1].bendOffset = 1;
            autoHand.fingers[2].bendOffset = 1;
            autoHand.fingers[3].bendOffset = 1;
            autoHand.fingers[4].bendOffset = 1;
        }
        public void setscissors()
        {
            // pos.setPose(new Vector3(0, 0, 3), new Vector3(0, 0, 0));

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
             // yield return new WaitForSeconds(1);
             // setRock();
             // yield return new WaitForSeconds(1);
             // setPaper();
             // yield return new WaitForSeconds(1);
             // setRock();
             // yield return new WaitForSeconds(1);
             // setscissors();
             //
             // yield return new WaitForSeconds(4);
             // setPaper();
             // yield return new WaitForSeconds(4);
             // setscissors();
             // yield return new WaitForSeconds(4);
             // setPaper();
             // yield return new WaitForSeconds(4);
             // setscissors();
             
             
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
        // public void changePosition()
        // {
        //     handPosition.position.x = 20f;
        // }
    }
}
