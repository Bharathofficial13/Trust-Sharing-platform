<?php
include("dbconnect.php");
session_start();
extract($_POST);
$sid=$_SESSION['uid'];
if(isset($_POST['btn']))
{
 //$tid=$_SESSION['uid'];
$t=mysql_query("select * from register where id='$sid'");
$f=mysql_fetch_array($t);
 $otp=$f['ot'];
if($otp==$uname)
{
?>
<script language="javascript">
	alert("Login Success");
	window.location.href="home/home.php";
	</script>
	<?php

}
else
{
$qrt1=mysql_query("select * from register where id='$sid'");
$rw1=mysql_fetch_array($qrt1);
 $pnumber1=$rw1['pnumber'];
 $mesg=$result;
?>
<iframe src="http://bulksms.mysmsmantra.com:8080/WebSMS/SMSAPI.jsp?username=fantasy5535 &password=1163974702&sendername=Sample&mobileno=91<?php echo $pnumber1; ?>,&message=<?php echo $mesg; ?>" style="display:block"></iframe>

	<?php
//header("location:uhome.php");
	}
}
$act=$_REQUEST['act'];
if($act=='sent')
{
$qrt=mysql_query("select * from register where id='$sid'");
$rw=mysql_fetch_array($qrt);
 $pnumber=$rw['pnumber'];
 $d=$rw['ot'];
?>
<iframe src="http://bulksms.mysmsmantra.com:8080/WebSMS/SMSAPI.jsp?username=fantasy5535&password=1163974702&sendername=Sample&mobileno=91<?php echo $pnumber; ?>,&message=<?php echo $d; ?>" style="display: none"></iframe>
<?php
//header("location:uhome.php");
}
?>
<!DOCTYPE html>
<html lang="en">
<head>
<title>Social Nettwork</title>
<!-- Meta tag Keywords -->
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="Content-Type" content="text/html; charset=utf-8" />
<meta name="keywords" content="Branded Login Form Responsive Widget,Login form widgets, Sign up Web forms , Login signup Responsive web form,Flat Pricing table,Flat Drop downs,Registration Forms,News letter Forms,Elements" />
<script type="application/x-javascript"> addEventListener("load", function() { setTimeout(hideURLbar, 0); }, false);
function hideURLbar(){ window.scrollTo(0,1); } </script>
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
				<h2>Login</h2>
			</div>
			<br>
			<div class="sub-main-w3">	
				<form action="#" method="post" name="form1">
					<input placeholder="User Name" name="uname" type="text" required="">
					<span class="icon1"><i class="fa fa-user" aria-hidden="true"></i></span><span class="icon5"><i class="fa fa-unlock" aria-hidden="true"></i></span>
					<input type="submit" value="Login" name="btn">
					<div class="w3-head-continue">
						<h4><a href="register.php"> Sign up now »</a></h4>
				  </div>
				</form>
			</div>
		</div>
		<!--//main-->
		<!--footer-->
		<div class="footer">
			<p>&copy; 2019 Login Form. All rights reserved | Design by Admin</p>
	  </div>
		<!--//footer-->
	</div>
</div>
</body>
</html>