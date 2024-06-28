<?php
include("dbconnect.php");
session_start();
extract($_POST);


$now =date("Y-m-d");


$keyy = substr(md5(time()), 0, 10);

class Block {
    public $index;
    public $previous_hash;
    public $current_hash;
    public $transaction;
    public $timestamp;
    public $nonce;

    public function __construct($index, $transaction, $previous_hash, $nonce) {
        if ($index !== null) {
            $this->index = $index;
        } else {
            $this->index = 1;
        }

        if ($previous_hash !== null) {
            $this->previous_hash = $previous_hash;
        } else {
            $this->previous_hash = str_repeat("0", 64);
        }

        $this->transaction = $transaction;
        $this->timestamp = time();
        $this->nonce = $nonce;
        $this->current_hash = $this->getHash();
    }

    public function getHash() {
        $data = array(
            $this->index,
            $this->previous_hash,
            $this->transaction,
            $this->nonce
        );

        $data = json_encode($data);
        $hash = hash('sha256', $data);
        return $hash;
    }
}

$block = new Block(1, 'data1', $keyy, 0);








 


 
$hashTest=mysql_query("select * from register");

 $num=mysql_num_rows($hashTest);

if($num<1){
$hash1='0';
$hash2=$block->current_hash;



}

elseif($num>0){
$hashTest1=mysql_query("select max(id) from register");
$row=mysql_fetch_array($hashTest1);
 $maxid=$row['max(id)'];
$hashTest2=mysql_query("select * from register where id=$maxid ");

$row1=mysql_fetch_array($hashTest2);
 $hash1=$row1['hash2'];

$hash2=$block->current_hash;







}

























if(isset($_POST['btn']))
{
$qr=mysql_query("select * from  register where  anumber='$anumber' || email='$email' || pnumber='$pnumber'");
$n=mysql_num_rows($qr);
if($n==0)
{
$max_qry = mysql_query("select max(id) from register");
	$max_row = mysql_fetch_array($max_qry); 
	$id=$max_row['max(id)']+1;
$qry=mysql_query("insert into register(id,name,email,pnumber,anumber,uname,password,dob,work,address,image,ot,hash1,hash2)values('$id','$name','$email','$pnumber','$anumber','$uname','$password','','','','','','$hash1','$hash2')");
header("location:register1.php?id=$id&name=$name");
}else{
?>
<script language="javascript">
	alert("fake profile");
	window.location.href="register.php";
	</script>
	<?php
}
}
?>

<!DOCTYPE html>
<html lang="en">
<head>
<title>Social Network</title>
<!-- Meta tag Keywords -->
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="Content-Type" content="text/html; charset=utf-8" />
<meta name="keywords" content="Branded Login Form Responsive Widget,Login form widgets, Sign up Web forms , Login signup Responsive web form,Flat Pricing table,Flat Drop downs,Registration Forms,News letter Forms,Elements" />
<script src="/vendor/intl-tel-input/build/js/intlTelInput.min.js"></script>
<script type="application/x-javascript"> addEventListener("load", function() { setTimeout(hideURLbar, 0); }, false);
function hideURLbar(){ window.scrollTo(0,1); } </script>
<script language="javascript">
            function validate()
            {
			   if (document.form1.name.value == "")
                {
                    alert("Enter the Name");
                    document.form1.name.focus();
                    return false;
                }
				var name=document.form1.name;
			    var letters = /^[A-Za-z]+$/;  
				if(name.value.match(letters))  
				{  
				return true;  
				}  
				else  
				{  
				alert('Username must have alphabet characters only');  
				document.form1.name.select();  
				return false;  
				}
				               
                if (document.form1.email.value == "")
                {
                    alert("Enter the E-mail");
                    document.form1.email.focus();
                    return false;
                }
				if (/^\w+([\.-]?\w+)*@\w+([\.-]?\w+)*(\.\w{2,3})+$/.test(document.form1.email.value))  
				  {  
					return true;  
				  }  
				  else
				  {
					alert("You have entered an invalid email address!");
					document.form1.email.select();
					return false; 
				  }
                if (document.form1.pnumber.value == "")
                {
                    alert("Enter the Contact No.");
                    document.form1.pnumber.focus();
                    return false;
                }
				if (isNaN(document.form1.pnumber.value))
                {
                    alert("Invalid Contact No.");
                    document.form1.pnumber.select();
                    return false;
                }
				if (document.form1.pnumber.value.length != 10)
                {
                    alert("10 digists only allowed!!");
                    document.form1.pnumber.select();
                    return false;
                }
				
				
                //finishMD();
                return true;
            }
        </script>
<!-- Meta tag Keywords -->
<!-- css files -->
<link rel="stylesheet" href="css/style.css" type="text/css" media="all" /> <!-- Style-CSS --> 
<link rel="stylesheet" href="css/font-awesome.css"> <!-- Font-Awesome-Icons-CSS -->
<!-- //css files -->
<!-- online-fonts -->
<link href="//fonts.googleapis.com/css?family=Raleway:100,100i,200,200i,300,300i,400,400i,500,500i,600,600i,700,700i,800,800i,900,900i" rel="stylesheet">
<!-- //online-fonts -->

</head>
<body>
<div class="w3-agile-banner">
	<div class="center-container">
			<div class="main-content-agile">
			<div class="wthree-pro">
				<h2>Sign up </h2>
			</div>
			<br>
			
			<div class="sub-main-w3">	
				<form action="#" method="post" name="form1">
					<div><input placeholder="Name" name="name" type="text" required="">
					
					</div>
							
				
				  <div><input placeholder="Email Address" required="" name="email" type="text" pattern="[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,3}$" title="enter a valid email id" >
										</div>
					<input placeholder="Phone Number" required="" pattern="[6789][0-9]{9}"  title="firstnumber should be start with 6 or 7 or 8 or 9 range 10" name="pnumber" type="text">
					<input placeholder="Aadhar Number" required=""     pattern="[0-9]{12}"  title="range 12" name="anumber" type="text" required>
										
					<input  placeholder="User Name" required="" name="uname" type="text" required="">
					<input  placeholder="Password" required="" name="password" type="password" required="">
					
					
					<div class="rem-w3">
					
						<span class="check-w3"><input type="checkbox" />Remember Me</span>
						<div class="clear"></div>
					</div>
					<input type="submit" value="Sign Up" name="btn"  >
					<div class="w3-head-continue">
						<h4><a href="index.php"> Back »</a></h4>
				  </div>
				</form>
			</div>
		</div>
		<!--//main-->
		<!--footer-->
		<div class="footer">
			<p>&copy;</p>
	  </div>
		<!--//footer-->
	</div>
</div>
</body>
</html>






