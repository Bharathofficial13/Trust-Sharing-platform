<?php
include("dbconnect.php");
session_start();
extract($_POST);
$uid=$_SESSION['uid'];
$fg=mysql_query("select * from register where id='$uid'");
$nm=mysql_fetch_array($fg);
$mn=$nm['name'];
ob_start();  
   //Get the ipconfig details using system commond  
   system('ipconfig /all');  
   // Capture the output into a variable  
   $mycomsys=ob_get_contents();  
   // Clean (erase) the output buffer  
   ob_clean();  
   $find_mac = "Physical"; //find the "Physical" & Find the position of Physical text  
   $pmac = strpos($mycomsys, $find_mac);  
   // Get Physical Address  
   $macaddress=substr($mycomsys,($pmac+36),17);  
   //Display Mac Address  
    $macaddress;
	$exec = exec("hostname"); 
$hostname = trim($exec);
$ip = gethostbyname($hostname);
$_SESSION['ipadd']=$ip;
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
			<?php
					$qry=mysql_query("select * from register where id='$uid'");
					$rw=mysql_fetch_array($qry); 
					?>
				
			</div>
			<div class="profile-usertitle">
				<div class="profile-usertitle-name"><?php echo $rw['name'];?></div>
				<div class="profile-usertitle-status"><span class="indicator label-success"></span>&nbsp;&nbsp;&nbsp;&nbsp;<?php echo $rw['name'];?></div>
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
			<div class="col-lg-12">
				<h1 class="page-header">Post List </h1>
			</div>
		</div><!--/.row-->
				
		
		<div class="row">
			<div class="col-lg-12">
				<div class="panel panel-default">
					<div class="panel-heading">List</div>
				  <div class="panel-body">
				     <?php
																	$qr=mysql_query("select * from frlist where id='$uid' && status='2'");
																	$ml=mysql_num_rows($qr);
																	if($ml>0)
																	{
																	while($row=mysql_fetch_array($qr))
																	{
																	$fid=$row['frid'];
																	
																	
																	
																	
																	
																	
																	
																	
																	
																	
																	$qt=mysql_query("select * from gpost where status='1'  ORDER BY id DESC");
																	
																	}
																	while($r=mysql_fetch_array($qt))
																	{
																	$rt=$r['sid'];
																	$g=mysql_query("select * from register where id='$rt'");
																	$ghh=mysql_fetch_array($g);
																	?>
          </h3>
                                                  <form action="" method="post" name="form">
                                                    <table width="91%" border="0">
                                                      <tr>
                                                        <td height="39">&nbsp;</td>
                                                        <td colspan="3"><span class="style2"><img src="images/<?php echo $ghh['image'];?>" width="25" height="25" class="im1">&nbsp;<?php echo $ghh['name'];?>&nbsp;&nbsp;&nbsp;<?php echo $r['type'];?></span></td>
                                                        <td></td>
                                                      </tr>
                                                      <tr>
                                                        <td height="32">&nbsp;</td>
                                                        <td colspan="3"><span class="style2">Date:<?php echo $r['date'];?> Time:<?php echo $r['time'];?></span></td>
                                                        <td>&nbsp;</td>
                                                      </tr>
                                                      <tr>
                                                        <td height="32">&nbsp;</td>
                                                        <td colspan="3"><div align="left" class="task-desc task-desc"><?php echo $r['cption'];?></div></td>
                                                        <td>&nbsp;</td>
                                                      </tr>
                                                      <tr>
                                                        <td width="16%" height="224">&nbsp;</td>
                                                        <td colspan="3" valign="top"><div align="left"><img src="image/<?php echo $r['image'];?>" width="250" height="200" ></div></td>
                                                        <td width="30%">&nbsp;</td>
                                                      </tr>
													   <tr>
                <td>&nbsp;</td>
                <td width="21%"><div align="center"><i class="fa fa-comments"></i><a href="comment.php?id=<?php echo $r['id'];?>">Comments</a></div></td>
               
                <td><i class="fa fa-share-square"></i><a href="image/<?php echo $r['image'];?>" download>Download</a></td>
              </tr>
                                                      <tr>
                                                        <td>&nbsp;</td>
                                                        
                                                        <td>&nbsp;</td>
                                                      </tr>
                                                    </table>
                                                    <?php
																	
																	}
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
