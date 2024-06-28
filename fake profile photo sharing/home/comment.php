<?php
include("dbconnect.php");
session_start();
extract($_POST);
$uid=$_SESSION['uid'];

  error_reporting(0);
$fg=mysql_query("select * from register where id='$uid'");
$nm=mysql_fetch_array($fg);
$mn=$nm['name'];
$cmid=$_REQUEST['id'];
if(isset($_POST['btn']))
{
$qst=mysql_query("select * from cmt where cmt='$commt'");
$y=mysql_num_rows($qst);
if($y>0)
{
$st=1;
$ts=1;
$nnn=mysql_query("select * from gpost where id='$cmid'");
$gj=mysql_fetch_array($nnn);
$aqq=$gj['sid'];
$mmb=mysql_query("select * from register where id='$aqq'");
$fo=mysql_fetch_array($mmb);
$pnumber=$fo['pnumber'];
$msg=$mn.', Commant post:'.$commt;
?>
<iframe src="http://bulksms.mysmsmantra.com:8080/WebSMS/SMSAPI.jsp?username=fantasy5535 &password=1163974702&sendername=Sample&mobileno=91<?php echo $pnumber; ?>,&message=<?php echo $msg; ?>" style="display:block"></iframe>

	<?php
}
else
{
$st=0;
$ts=0;
}
$max_qry = mysql_query("select max(id) from comment");
	$max_row = mysql_fetch_array($max_qry); 
	$id=$max_row['max(id)']+1;
	$date=date("d-m-y");
	$t=date("h:i:sa");
	

$csvFile = 'negative words.csv';


if (!file_exists($csvFile)) {
    die("File not found.");
}

$forbiddenWords = array();

$file = fopen($csvFile, 'r');

while (($line = fgetcsv($file)) !== false) {
   
     $forbiddenWords[] = $line[0];
	 
	 
	 
	 
	 
}


fclose($file);




$variable = $commt; 



foreach ($forbiddenWords as $word) {
    if (strpos($variable, $word) !== false) {
        
        $st = 'bad';
        break; 
    }
}


echo $st;


	
	
	
	
	
	$qrtml=mysql_query("insert into comment values('$id','$cmid','$uid','$commt','$st','$ts','$date','$t')");
	if($qrtml)
	{
	//header("location:comment.php?id=".$cmid);
	echo "<script language='javascript'>
//	alert('Commant  Sent');
//	window.location.href='comment.php?id=$cmid';
	</script>";
	}
	else
	{
	echo "<script language='javascript'>
	alert('Commant Sent');
	window.location.href='comment.php?id=$cmid';
	</script>";
	}
}
?>
<?php
$bl=mysql_query("select * from gpost");
while($bll=mysql_fetch_array($bl))
{
$bid=$bll['sid'];
$mmid=$bll['id'];
$qr=mysql_query("select * from comment where mid='$mmid' && status='1'");
$sd=mysql_num_rows($qr);
if($sd==4)
{
$fg=mysql_fetch_array($qr);
$bbid=$fg['sid'];
$rt=mysql_query("update frlist set status='3' where id='$bid' && frid='$bbid' && status='2'");
$rt;
$dt=mysql_query("update frlist set status='3' where id='$bbid' && frid='$bid' && status='2'");
$dt;
}
}
?>
<!DOCTYPE html>
<html>
<head>
	<meta charset="utf-8">
	<meta name="viewport" content="width=device-width, initial-scale=1">
	<title>Facebook Home</title>
	<link href="css/bootstrap.min.css" rel="stylesheet">
	<link href="css/font-awesome.min.css" rel="stylesheet">
	<link href="css/datepicker3.css" rel="stylesheet">
	<link href="css/styles.css" rel="stylesheet">
	
	<!--Custom Font-->
	<link href="https://fonts.googleapis.com/css?family=Montserrat:300,300i,400,400i,500,500i,600,600i,700,700i" rel="stylesheet">
	<!--[if lt IE 9]>
	<script src="js/html5shiv.js"></script>
	<script src="js/respond.min.js"></script>
	<![endif]-->
    <style type="text/css">
<!--
.im {border-radius: 50%;
}
.im1 {  border-radius: 50%;
}
.im2 {  border-radius: 50%;
}
.style1 {color: #000000}
-->
    </style>
</head>
<body>
	<nav class="navbar navbar-custom navbar-fixed-top" role="navigation">
		<div class="container-fluid">
			<div class="navbar-header">
				<button type="button" class="navbar-toggle collapsed" data-toggle="collapse" data-target="#sidebar-collapse"><span class="sr-only">Toggle navigation</span>
					<span class="icon-bar"></span>
					<span class="icon-bar"></span>
					<span class="icon-bar"></span></button>
				<a class="navbar-brand" href="#"><span>Facebook</span>Home</a>
				<ul class="nav navbar-top-links navbar-right">
				</ul>
			</div>
		</div><!-- /.container-fluid -->
	</nav>
	<div id="sidebar-collapse" class="col-sm-3 col-lg-2 sidebar">
		<div class="profile-sidebar">
			<div class="profile-userpic">
			
			</div>
			<div class="profile-usertitle">
				<div class="profile-usertitle-name"><?php echo $rw['name'];?></div>
				<div class="profile-usertitle-status"><span class="indicator label-success"></span>Online</div>
			</div>
			<div class="clear"></div>
		</div>
		<div class="divider"></div>
		<form role="search">
			<div class="form-group">
				<input type="text" class="form-control" placeholder="Search">
			</div>
		</form>
		<ul class="nav menu">
		<li><a href="home.php"><i class="fa fa-home"></i> <span>Home</span></a></li>
										 <li><a href="frlist.php"><i class="fa fa-book"></i> <span>Friend List</span></a></li>
										<li><a href="list.php"><i class="fa fa-users"></i> <span>List</span></a></li>
									<li><a href="post.php"><i class=" fa fa-book"></i> <span>Post</span></a></li>
									<li><a href="accept.php"><i class="fa fa-users"></i> <span>Accept</span></a></li>
									<li><a href="group.php"><i class="fa fa-users"></i> <span>Create Group</span></a></li>
									<li><a href="grouppost.php"><i class="fa fa-users"></i> <span>Group Post</span></a></li>
									
									<li><a href="profile.php"><i class="fa fa-bell-o"></i> <span>Profile</span></a></li>
									<li><a href="index.php"><i class="fa fa-power-switch"></i> <span>Logout</span></a></li>
		</ul>
	</div><!--/.sidebar-->
		
	<div class="col-sm-9 col-sm-offset-3 col-lg-10 col-lg-offset-2 main">
		<div class="row">
			<ol class="breadcrumb">
				<li><a href="#">
					<em class="fa fa-home"></em>
				</a></li>
				<li class="active">Forms</li>
			</ol>
		</div><!--/.row-->
		
		<div class="row">
		  <div class="col-lg-12"></div>
		</div><!--/.row-->
				
		
		<div class="row">
			<div class="col-lg-12">
				<div class="panel panel-default">
					<div class="panel-heading"></div>
				  <div class="panel-body">
				     <form action="" method="post" name="form">
																	<?php
												$qr=mysql_query("select * from gpost where id='$cmid'");
																	while($row=mysql_fetch_array($qr))
																	{
																	$usid=$row['sid'];																
																														$g=mysql_query("select * from register where id='$usid'");
																	$ghh=mysql_fetch_array($g);
																	?>
																	<table width="91%" border="0">
                                                                      <tr>
                                                                        <td height="39">&nbsp;</td>
                                                                        <td width="54%"><img src="images/<?php echo $ghh['image'];?>" width="25" height="25" class="im">&nbsp;<?php echo $ghh['name'];?></td>
                                                                        <td>&nbsp;</td>
                                                                      </tr>
                                                                      <tr>
                                                                        <td height="32">&nbsp;</td>
                                                                        <td><div align="left"><?php echo $r['cption'];?></div></td>
                                                                        <td>&nbsp;</td>
                                                                      </tr>
                                                                      <tr>
                                                                        <td width="16%" height="317">&nbsp;</td>
                                                                        <td valign="top"><div align="left"><img src="image/<?php echo $row['image'];?>" width="250" height="300" ></div></td>
                                                                        <td width="30%">&nbsp;</td>
                                                                      </tr>
                                                                      <tr>
                                                                        <td height="77">&nbsp;</td>
                                                                        <td valign="top"><label>
                                                                          `
                                                                              <textarea name="commt" cols="30" rows="4" id="commt"></textarea>
                                                                        <input type="submit" name="btn" value="Submit">
                                                                        </label></td>
                                                                        <td>&nbsp;</td>
                                                                      </tr>
                                                                      <tr>
                                                                        <td height="70">&nbsp;</td>
                                                                        <td colspan="2" valign="top">
													<div class="vertical-menu">
                                                    <table width="80%" border="0">
                                                    <?php
													$ct=mysql_query("select * from comment where mid='$cmid' && status='0'");
													while($cm=mysql_fetch_array($ct))
													{ 
													$ssid=$cm['sid'];
													$ml=mysql_query("select * from register where id='$ssid'");
													$cmr=mysql_fetch_array($ml);
													?>
													  <tr>
													    <td>&nbsp;</td>
													    <td>&nbsp;</td>
												      </tr>
													  <tr>
                                                      <td width="12%" height="34"><img src="images/<?php echo $cmr['image'];?>" width="25" height="25" class="im">&nbsp;</td>
                                                      <td width="88%"><?php echo $cmr['name'];?></td>
                                                    </tr>
													  <tr>
													    <td height="45">&nbsp;</td>
													    <td><?php echo $cm['date'];?>-<?php echo $cm['time'];?></td>
												      </tr>
													  <tr>
													    <td height="45">&nbsp;</td>
													    <td><?php echo $cm['cmt'];?>...</td>
												      </tr>
                                                    <?php
													}
													?>
													</table>
  
</div></td>
                                                                      </tr>
                                                                    </table>
																	<?php
																	
																	}
																	?>
																	</form>
		  
                                          
				  </div>
				</div><!-- /.panel-->
			</div>
			<!-- /.panel-->
	  </div><!-- /.col-->
			<div class="col-sm-12">
				<p class="back-link">Lumino Theme by <a href="https://www.medialoot.com">Medialoot</a></p>
			</div>
</div><!-- /.row -->
	</div><!--/.main-->
	
<script src="js/jquery-1.11.1.min.js"></script>
	<script src="js/bootstrap.min.js"></script>
	<script src="js/chart.min.js"></script>
	<script src="js/chart-data.js"></script>
	<script src="js/easypiechart.js"></script>
	<script src="js/easypiechart-data.js"></script>
	<script src="js/bootstrap-datepicker.js"></script>
	<script src="js/custom.js"></script>
	
</body>
</html>
